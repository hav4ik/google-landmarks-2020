import os
from argparse import ArgumentParser
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers as keras_optimizers
import wandb
from wandb.keras import WandbCallback

from glrec.train import dataflow
from glrec.train import augmentation
from glrec.train import lr_schedulers
from glrec.train import utils as train_utils
from glrec.train.constants import constants as train_constants
from glrec.utils import log, StopWatch
from delg_model import DelgModel
from callbacks import GsutilRsync
from callbacks import get_callback


argument_parser = ArgumentParser(
        description='Training script for Google Landmarks Challenge 2020')
argument_parser.add_argument('yaml_path', type=str,
                             help='Path to the YAML experiment config file')
argument_parser.add_argument('-d', '--debug', action='store_true',
                             help='Whether to run in debug mode')


def load_dataset(dataset_config):
    """Loads training and validation datasets from YAML configuration"""

    # Enable parallel streams from multiple TFRecord files for TPU
    # (more on https://www.kaggle.com/docs/tpu)
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    # Load the training tfrecords
    train_filenames = dataflow.retrieve_tfrecord_filenames(
            **dataset_config['train_tfrecords'])
    train_dataset = tf.data.TFRecordDataset(
            train_filenames,
            num_parallel_reads=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.with_options(ignore_order)

    # Load the validation tfrecords
    validation_filenames = dataflow.retrieve_tfrecord_filenames(
            **dataset_config['validation_tfrecords'])
    validation_dataset = tf.data.TFRecordDataset(
            validation_filenames,
            num_parallel_reads=tf.data.experimental.AUTOTUNE)
    validation_dataset = validation_dataset.with_options(ignore_order)

    # Shuffle training set
    if 'train_shuffle' in dataset_config:
        train_dataset = train_dataset.repeat().shuffle(
                **dataset_config['train_shuffle'])

    # Parse dataset
    train_dataset = train_dataset.map(
            dataflow.get_gld_parser(
                image_size=dataset_config['image_size'],
                imagenet_augmentation=dataset_config['imagenet_crop']),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    validation_dataset = validation_dataset.map(
            dataflow.get_gld_parser(
                image_size=dataset_config['image_size'],
                imagenet_augmentation=dataset_config['imagenet_crop']),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return train_dataset, validation_dataset


def get_augmented_dataset(train_dataset, dataset_config):
    """Perform data augmentations, if any"""
    if 'train_augmentations' not in dataset_config:
        return train_dataset

    for item in dataset_config['train_augmentations']:
        class_name = item['class']
        kwargs = item['kwargs']
        transformer = getattr(augmentation, class_name)(**kwargs)
        train_dataset = train_dataset.map(
                lambda image, label: (transformer.transform(image), label))
    return train_dataset


def get_optimizer(algorithm, kwargs):
    """Loads the specified optimizer from Keras with given kwargs"""
    return getattr(keras_optimizers, algorithm)(**kwargs)


def prepare_dirs(storage_root, experiment_name):
    if storage_root[:5] == 'gs://':
        log.debug('Using GCS to store models. Employ the `rsync` strategy')
        storage_root = '/tmp/glc/'
    else:
        log.debug('Using local storage')
        storage_root = os.path.expanduser(storage_root)
    experiment_dir = os.path.join(storage_root, experiment_name)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    tensorboard_dir = os.path.join(experiment_dir, 'tensorboard')

    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.isdir(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    return experiment_dir, checkpoints_dir, tensorboard_dir


def calculate_class_weights(label_mapping_df, training_config):
    class_weights = None
    if 'class_weights' in training_config and label_mapping_df is not None:
        if training_config['class_weights'] == 'inv_log':
            log.debug('Class weights is set to 1/log(class count)')
            class_weights = np.array(label_mapping_df['num_samples'])
            class_weights = class_weights.astype(np.float64)
            class_weights = 1. / np.log(class_weights)
            mean_weight = np.mean(class_weights)
            weight_scale = 1. / mean_weight
            class_weights = class_weights * weight_scale
        else:
            raise ValueError('Only `inv_log` is supported right now.')
    return class_weights


def train_delg(experiment,
               glc_config,
               dataset_config,
               model_config,
               training_config,
               debug_mode=False):
    """Grand Training Loop for Google Landmarks Challenge 2020
    """
    wandb.init(project='google-landmarks-recognition-2020',
               name=experiment['name'],
               group=model_config['backbone_config']['architecture'],
               config={
                    'experiment': experiment,
                    'glc_config': glc_config,
                    'dataset_config': dataset_config,
                    'model_config': model_config,
                    'training_config': training_config,
               },
               id=experiment['name'],
               resume=True,
               allow_val_change=True)

    log.info('Started Experiment: ' + experiment['name'])
    log.info('All data will be saved to: ' + experiment['storage'])
    log.info('Experiment description: ' + experiment['description'])

    # ------------------------------------------------------------
    #   PREPARE ENVIRONMENT AND CONFIGURATIONS
    # ------------------------------------------------------------

    # Loading distribution strategy for TPU, CPU, and GPU
    distribution_strategy = train_utils.get_distribution_strategy()
    train_constants.set_mode(glc_config['gld_version'])

    # Resolve all scalars in configurations to adjust them for TPU
    dataset_config = train_utils.resolve_scalars_for_tpu(
            dataset_config, distribution_strategy.num_replicas_in_sync)
    model_config = train_utils.resolve_scalars_for_tpu(
            model_config, distribution_strategy.num_replicas_in_sync)
    training_config = train_utils.resolve_scalars_for_tpu(
            training_config, distribution_strategy.num_replicas_in_sync)

    # Directories to store models and logs
    experiment_dir, checkpoints_dir, tensorboard_dir = \
        prepare_dirs(experiment['storage'], experiment['name'])
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        yaml.dump({
            'experiment': experiment,
            'glc_config': glc_config,
            'dataset_config': dataset_config,
            'model_config': model_config,
            'training_config': training_config,
        }, f, default_flow_style=False)

    # Class ID mapping and counts
    label_mapping_df = None
    if 'gld_id_mapping' in glc_config and \
            glc_config['gld_id_mapping'] is not None:
        log.debug('Found `gld_id_mapping` file')
        label_mapping_csv_path = train_utils.resolve_file_path(
                glc_config['gld_id_mapping'])
        if os.path.isfile(label_mapping_csv_path):
            label_mapping_df = pd.read_csv(label_mapping_csv_path)
        else:
            raise RuntimeError('Failed to load `label_mapping_df`')

    # ------------------------------------------------------------
    #   PREPARE TRAINING AND TESTING DATASETS
    # ------------------------------------------------------------

    # Load training and validation datasets
    train_dataset, validation_dataset = load_dataset(dataset_config)

    # Data echoing to train faster https://arxiv.org/pdf/1907.05550.pdf
    def data_echoing(factor):
        return lambda image, label: \
            tf.data.Dataset.from_tensors((image, label)).repeat(factor)

    if 'data_echoing' in dataset_config:
        if dataset_config['data_echoing']['factor'] > 1 and \
                dataset_config['data_echoing']['when'] == 'before_aug':
            train_dataset = train_dataset.flat_map(
                    data_echoing(dataset_config['data_echoing']['factor']))

    # Data Augmentation
    train_dataset = get_augmented_dataset(train_dataset, dataset_config)

    # Data echoing to train faster https://arxiv.org/pdf/1907.05550.pdf
    if 'data_echoing' in dataset_config:
        if dataset_config['data_echoing']['factor'] > 1 and \
                dataset_config['data_echoing']['when'] == 'after_aug':
            train_dataset = train_dataset.flat_map(
                    data_echoing(dataset_config['data_echoing']['factor']))

    # Map labels if necessary (for gld-v2-clean)
    if label_mapping_df is not None:
        table = tf.lookup.StaticHashTable(
             tf.lookup.KeyValueTensorInitializer(
                 keys=tf.constant(
                     np.array(label_mapping_df['landmark_id']),
                     dtype=tf.int64),
                 values=tf.constant(
                     np.array(label_mapping_df['squeezed_id']),
                     dtype=tf.int64)),
             default_value=np.argmax(np.array(
                 label_mapping_df['num_samples'])))

        def label_mapping_func(image, label):
            return image, table.lookup(label)

        train_dataset = train_dataset.map(
                label_mapping_func,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_dataset = validation_dataset.map(
                label_mapping_func,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Form into batches, depending on number of replicas
    batch_size = training_config['batch_size']
    log.info(f'Batch size (adjusted for number of replicas): {batch_size}')
    epochs = training_config['epochs']
    samples_per_epoch = training_config['samples_per_epoch']

    # Limit batch size and dataset sizes in debug mode
    if debug_mode:
        log.warning('Debug mode is enabled. You can disable it in '
                    '"experiment" settings')

        batch_size = 2 * distribution_strategy.num_replicas_in_sync
        log.warning(f'Batch size is limited to {batch_size} in debug mode.')

        train_dataset = train_dataset.take(2 * batch_size)
        validation_dataset = validation_dataset.take(2 * batch_size)
        log.warning(f'Train and Val datasets limited to {2 * batch_size} '
                    'samples in debug mode.')

        epochs, samples_per_epoch = 2, 2 * batch_size
        log.warning('Number of epochs is limited to 2 in debug mode.')

    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

    validation_dataset = validation_dataset.batch(batch_size)
    validation_dataset = validation_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

    # Training loop. Datasets are adjusted to Keras format
    def to_keras_format(image, label):
        label = tf.reshape(tf.squeeze(label), [batch_size, ])
        image = tf.reshape(image, [batch_size, *dataset_config['image_size']])
        return (image, label), label

    train_dataset = train_dataset.map(
            to_keras_format,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    validation_dataset = validation_dataset.map(
            to_keras_format,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # ------------------------------------------------------------
    #   PREPARE THE MODEL
    # ------------------------------------------------------------

    # Load and compile model with appropriate loss function
    with distribution_strategy.scope(), StopWatch('Model compiled in:'):

        # Load the model
        model = DelgModel(**model_config)

        # Classification loss function
        class_weights = calculate_class_weights(
                label_mapping_df, training_config)
        if class_weights is not None:
            class_weights = tf.convert_to_tensor(class_weights)
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                    reduction=tf.keras.losses.Reduction.NONE)

            def classification_loss_func(y_true, y_pred):
                y_true = tf.cast(y_true, tf.int32)
                sample_weights = tf.gather(class_weights, y_true)
                per_sample_loss = loss_object(y_true, y_pred,
                                              sample_weight=sample_weights)
                return tf.nn.compute_average_loss(
                    per_sample_loss, global_batch_size=batch_size)
        else:
            classification_loss_func = 'sparse_categorical_crossentropy'

        # Reconstruction loss function (the reconstruction score itself is
        # already calculated in the autoencoder layer itself.
        def reconstruction_loss_func(y_true, y_pred):
            return tf.nn.compute_average_loss(
                    y_pred, global_batch_size=batch_size)

        # Prepare the losses and metrics for each model's outputs (heads).
        # The order of model's outputs, by training mode, is as follows:
        # - global_only:
        #     - global_output
        # - local_only:
        #     - local_classification_output
        #     - local_reconstruction_score
        # - local_and_global:
        #     - global_output
        #     - local_classification_output
        #     - local_reconstruction_score
        if model_config['training_mode'] == 'global_only':
            losses = classification_loss_func
            metrics = ['sparse_categorical_accuracy']
            loss_weights = None
        elif model_config['training_mode'] == 'local_only':
            losses = [classification_loss_func,
                      reconstruction_loss_func]
            metrics = [['sparse_categorical_accuracy'], []]
            loss_weights = [training_config['attention_weight'],
                            training_config['reconstruction_weight']]
        elif model_config['training_mode'] == 'local_and_global':
            losses = [classification_loss_func,
                      classification_loss_func,
                      reconstruction_loss_func]
            metrics = [['sparse_categorical_accuracy'],
                       ['sparse_categorical_accuracy'],
                       []]
            loss_weights = [1.0,
                            training_config['attention_weight'],
                            training_config['reconstruction_weight']]

        else:
            raise RuntimeError('training_mode should be either global_only, '
                               'local_only, or local_and_global.')

        model.compile(
                optimizer=get_optimizer(**training_config['optimizer']),
                loss=losses, metrics=metrics, loss_weights=loss_weights)

        # Building model by calling. This is necessary for some weird stuff
        # that I can't fully explain.
        dummy_input_images = tf.convert_to_tensor(
                np.random.rand(batch_size, *dataset_config['image_size']),
                dtype=tf.float32)
        dummy_true_labels = tf.convert_to_tensor(
                np.random.randint(0, 1337, size=(batch_size, 1)),
                dtype=tf.int32)
        _ = model([dummy_input_images, dummy_true_labels],
                  first_time_warmup=True)


        # Just calling model.build won't call building of child layers.
        # model.build([
        #     [batch_size, *dataset_config['image_size']],
        #     [batch_size, ],
        # ])

        if 'from_checkpoint' in training_config:
            # `by_name` flag is used if we're loading from a different
            # architecture, with some layers in common.
            previous_weights = train_utils.resolve_file_path(
                    training_config['from_checkpoint']['weights'])
            model.load_weights(
                    previous_weights,
                    by_name=training_config['from_checkpoint']['by_name'],
                    skip_mismatch=training_config[
                        'from_checkpoint']['skip_mismatch'])

    # ------------------------------------------------------------
    #   PREPARE TRAINING CALLBACKS
    # ------------------------------------------------------------

    # ModelCheckpoint callback
    checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoints_dir,
                     '{epoch:03d}_val_loss={val_loss:.5f}.hdf5'),
        save_freq='epoch')

    # TensorBoard callback
    if experiment['storage'][:5] == 'gs://':
        # This is necessary for TPU training
        tb_log_dir = os.path.join(
                experiment['storage'], experiment['name'], 'tensorboard')
        log.debug('Setting TensorBoard to write to GCS bucket.')
    else:
        tb_log_dir = tensorboard_dir

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tb_log_dir,
            write_graph=False,
            update_freq='epoch')

    training_callbacks = [checkpoints_callback,
                          tensorboard_callback,
                          WandbCallback(save_model=False)]

    # Learning Rate Scheduler
    if 'lr_scheduler' in training_config:
        lr_scheduler_config = training_config['lr_scheduler']
        if not hasattr(lr_schedulers, lr_scheduler_config['type']):
            raise ValueError('The specified lr_scheduler is not supported')
        lr_scheduler = getattr(lr_schedulers, lr_scheduler_config['type'])(
                **lr_scheduler_config['kwargs'])
        training_callbacks.append(lr_scheduler)

    # Additional callbacks required from the config
    for callback_config_item in training_config['additional_callbacks']:
        training_callbacks.append(get_callback(**callback_config_item))

    # If storing on GCS, call `gsutil rsync` after each epoch.
    # This is added as last callback, since it is supposed to be
    # called after model and tensorboard saving.
    if experiment['storage'][:5] == 'gs://':
        gsutil_rsync_callback = GsutilRsync(
                experiment_dir,
                os.path.join(experiment['storage'], experiment['name']))
        training_callbacks.append(gsutil_rsync_callback)

    # ------------------------------------------------------------
    #   MODEL TRAINING
    # ------------------------------------------------------------

    if 'initial_epoch' in training_config:
        initial_epoch = training_config['initial_epoch']
        log.debug(f'Initial epoch is set to {initial_epoch}')
    else:
        initial_epoch = 0

    model.fit(train_dataset,
              steps_per_epoch=samples_per_epoch // batch_size,
              epochs=epochs,
              callbacks=training_callbacks,
              validation_data=validation_dataset,
              initial_epoch=initial_epoch,
              verbose=training_config['verbose'])


if __name__ == '__main__':
    args = argument_parser.parse_args()
    with open(args.yaml_path, 'r') as f:
        train_delg(debug_mode=args.debug, **yaml.safe_load(f.read()))

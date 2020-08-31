import os
from argparse import ArgumentParser
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers as keras_optimizers

from glrec.train import dataflow
from glrec.train import augmentation
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
        storage_root = '/tmp/glc/'
    else:
        storage_root = os.path.expanduser(storage_root)
    experiment_dir = os.path.join(storage_root, experiment_name)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    tensorboard_dir = os.path.join(experiment_dir, 'tensorboard')

    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.isdir(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    return experiment_dir, checkpoints_dir, tensorboard_dir


def train_delg(experiment,
               glc_config,
               dataset_config,
               model_config,
               training_config):
    """Grand Training Loop for Google Landmarks Challenge 2020
    """
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

    # Class ID mapping and counts
    label_mapping_df = None
    if 'gld_id_mapping' in glc_config and \
            glc_config['gld_id_mapping'] is not None:
        label_mapping_csv_path = train_utils.resolve_file_path(
                glc_config['gld_id_mapping'])
        if os.path.isfile(label_mapping_csv_path):
            label_mapping_df = pd.read_csv(label_mapping_csv_path)

    # ------------------------------------------------------------
    #   PREPARE TRAINING AND TESTING DATASETS
    # ------------------------------------------------------------

    # Load training and validation datasets
    train_dataset, validation_dataset = load_dataset(dataset_config)
    train_dataset = get_augmented_dataset(train_dataset, dataset_config)

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
    if experiment['mode'] == 'debug':
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
        label = tf.squeeze(label)
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

    # Load and compile model
    with distribution_strategy.scope(), StopWatch('Model compiled in:'):
        model = DelgModel(**model_config)
        model.compile(
                optimizer=get_optimizer(**training_config['optimizer']),
                loss=['sparse_categorical_crossentropy'],
                metrics=['sparse_categorical_accuracy'])

    # ------------------------------------------------------------
    #   PREPARE TRAINING CALLBACKS
    # ------------------------------------------------------------

    # Tensorboard and ModelCheckpoint callbacks
    checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoints_dir,
                     '{epoch:03d}_val_loss={val_loss:.5f}.hdf5'),
        save_freq='epoch')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=False,
            update_freq='epoch',
            profile_batch=2)
    training_callbacks = [checkpoints_callback, tensorboard_callback]

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

    model.fit(train_dataset,
              steps_per_epoch=samples_per_epoch // batch_size,
              epochs=epochs,
              callbacks=training_callbacks,
              validation_data=validation_dataset,
              verbose=1)


if __name__ == '__main__':
    with open(argument_parser.parse_args().yaml_path, 'r') as f:
        train_delg(**yaml.safe_load(f.read()))

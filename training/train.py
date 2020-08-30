from argparse import ArgumentParser
import yaml
import tensorflow as tf
from tensorflow.keras import optimizers as keras_optimizers

from glrec.train import dataflow
from glrec.train import augmentation
from glrec.train import utils as train_utils
from glrec.train.constants import constants as train_constants
from glrec.utils import log, StopWatch
from delg_model import DelgModel


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


def train(experiment,
          gld_version,
          dataset_config,
          model_config,
          training_config):
    """Grand Training Loop for Google Landmarks Challenge 2020
    """
    log.info('Started Experiment: ' + experiment['name'])
    log.info('All data will be saved to: ' + experiment['storage'])
    log.info('Experiment description: ' + experiment['description'])

    # Loading distribution strategy for TPU, CPU, and GPU
    distribution_strategy = train_utils.get_distribution_strategy()
    num_replicas = distribution_strategy.num_replicas_in_sync
    train_constants.set_mode(gld_version)

    # Load training and validation datasets
    train_dataset, validation_dataset = load_dataset(dataset_config)
    train_dataset = get_augmented_dataset(train_dataset, dataset_config)

    # Form into batches, depending on number of replicas
    batch_size = training_config['batch_size'] * num_replicas
    log.info(f'Batch size (adjusted for number of replicas): {batch_size}')
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(
            tf.data.experimental.AUTOTUNE)
    validation_dataset = validation_dataset.batch(batch_size)
    validation_dataset = validation_dataset.prefetch(
            tf.data.experimental.AUTOTUNE)

    # Load and compile model
    with distribution_strategy.scope(), StopWatch('Model compiled in:'):
        model = DelgModel(**model_config)
        model.compile(
            optimizer=get_optimizer(**training_config['optimizer']),
            loss=['sparse_categorical_crossentropy'],
            metrics=['sparse_categorical_accuracy'])

    # Training loop


if __name__ == '__main__':
    with open(argument_parser.parse_args().yaml_path, 'r') as f:
        train(**yaml.safe_load(f.read()))

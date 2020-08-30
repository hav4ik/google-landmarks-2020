# import os
# from argparse import ArgumentParser
import yaml
import tensorflow as tf

from glrec.train import dataflow
from glrec.train import augmentation
# from glrec.train import layers
from glrec.train import utils as train_utils
# from glrec.utils import log


def load_dataset(dataset_config):
    """Loads training and validation datasets from YAML configuration"""

    # Load the training tfrecords
    train_filenames = dataflow.retrieve_tfrecord_filenames(
            **dataset_config['train_tfrecords'])
    train_dataset = tf.data.TFRecordDataset(
            train_filenames,
            num_parallel_reads=tf.data.experimental.AUTOTUNE)

    # Shuffle training set
    if 'train_shuffle' in dataset_config:
        train_dataset = train_dataset.repeat().shuffle(
                **dataset_config['train_shuffle'])

    # Load the validation tfrecords
    validation_filenames = dataflow.retrieve_tfrecord_filenames(
            **dataset_config['validation_tfrecords'])
    validation_dataset = tf.data.TFRecordDataset(
            validation_filenames,
            num_parallel_reads=tf.data.experimental.AUTOTUNE)

    # Parse dataset
    train_dataset = train_dataset.map(dataflow.get_gld_parser(
            image_size=dataset_config['image_size'],
            imagenet_augmentation=dataset_config['imagenet_crop']))
    validation_dataset = validation_dataset.map(dataflow.get_gld_parser(
            image_size=dataset_config['image_size'],
            imagenet_augmentation=dataset_config['imagenet_crop']))
    return train_dataset, validation_dataset


def get_augmented(train_dataset, dataset_config):
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


def construct_backbone(model_config):
    """Loads a backbone network"""


def train(dataset_config):
    """Grand Training Loop for Google Landmarks Challenge 2020"""

    # Distribution strategy for TPU, GPU, and CPU
    distribution_strategy = train_utils.get_distribution_strategy()

    # Load training and validation datasets
    train_dataset, validation_dataset = load_dataset(dataset_config)

    # apply image augmentations during training, if any
    train_dataset = get_augmented(train_dataset, dataset_config)


if __name__ == '__main__':
    yaml_path = '../experiments/mobilenetv2_224x224_512_arcface.yaml'
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f.read())
        train(**config)

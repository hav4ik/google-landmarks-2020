import os
import csv
import tensorflow as tf
import cv2
from glrec import const


def to_hex(image_id) -> str:
    """Given Image Id, return its hex value"""
    return '{0:0{1}x}'.format(image_id, 16)


def get_image_path(subset, image_id):
    name = to_hex(image_id)
    return os.path.join(
            const.DATASET_DIR, subset, name[0], name[1], name[2],
            '{}.jpg'.format(name))


def image_path_to_id(image_path):
    return int(image_path.name.split('.')[0], 16)


def load_labelmap():
    with open(const.TRAIN_LABELMAP_PATH, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        labelmap = {row['id']: row['landmark_id'] for row in csv_reader}
    return labelmap


def load_image_tensor_from_path(image_path):
    """Loads an image from given path to a tensor"""
    return tf.convert_to_tensor(
        cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))


def is_public_dataset(labelmap):
    num_training_images = len(labelmap.keys())
    if num_training_images == const.NUM_PUBLIC_TRAIN_IMAGES:
        return True
    else:
        return False

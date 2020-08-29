import os
import tensorflow as tf
import functools
from tensorflow.io import FixedLenFeature


class _DataAugmentationParams(object):
    """Default parameters for augmentation."""
    # The following are used for training.
    min_object_covered = 0.1
    aspect_ratio_range_min = 3. / 4
    aspect_ratio_range_max = 4. / 3
    area_range_min = 0.08
    area_range_max = 1.0
    max_attempts = 100
    update_labels = False
    # 'central_fraction' is used for central crop in inference.
    central_fraction = 0.875

    random_reflection = False
    input_rows = 321
    input_cols = 321


def _normalize_images(images, pixel_value_scale=0.5, pixel_value_offset=0.5):
    """Normalize pixel values in image.
    Output is computed as
    normalized_images = (images - pixel_value_offset) / pixel_value_scale.
    Args:
      images: `Tensor`, images to normalize.
      pixel_value_scale: float, scale.
      pixel_value_offset: float, offset.
    Returns:
      normalized_images: `Tensor`, normalized images.
    """
    images = tf.cast(images, tf.float32)
    normalized_images = tf.math.divide(
        tf.subtract(images, pixel_value_offset), pixel_value_scale)
    return normalized_images


def _imagenet_crop(image):
    """Imagenet-style crop with random bbox and aspect ratio.
    Args:
      image: a `Tensor`, image to crop.
    Returns:
      cropped_image: `Tensor`, cropped image.
    """
    params = _DataAugmentationParams()
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    (bbox_begin, bbox_size, _) = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=params.min_object_covered,
            aspect_ratio_range=(params.aspect_ratio_range_min,
                                params.aspect_ratio_range_max),
            area_range=(params.area_range_min, params.area_range_max),
            max_attempts=params.max_attempts,
            use_image_if_no_bounding_boxes=True)
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    cropped_image.set_shape([None, None, 3])

    cropped_image = tf.image.resize(
            cropped_image,
            [params.input_rows, params.input_cols],
            method='area')
    if params.random_reflection:
        cropped_image = tf.image.random_flip_left_right(cropped_image)

    return cropped_image


def _parse_function(example, name_to_features, image_size, augmentation):
    """
    Parse a single TFExample to get the image and label and process the image.
    Args:
      example: a `TFExample`.
      name_to_features: a `dict`. The mapping from feature names to its type.
      image_size: an `int`. The image size for the decoded image, on each side.
      augmentation: a `boolean`. True if the image will be augmented.
    Returns:
      image: a `Tensor`. The processed image.
      label: a `Tensor`. The ground-truth label.
    """
    parsed_example = tf.io.parse_single_example(example, name_to_features)
    # Parse to get image.
    image = parsed_example['image/encoded']
    image = tf.io.decode_jpeg(image)
    image = _normalize_images(
            image, pixel_value_scale=128.0, pixel_value_offset=128.0)
    if augmentation:
        image = _imagenet_crop(image)
    else:
        image = tf.image.resize(image, [image_size[0], image_size[1]])
        image.set_shape([image_size[0], image_size[1], 3])
    # Parse to get label.
    label = parsed_example['image/class/label']

    return image, label


def get_gld_parser(image_size=(512, 512), imagenet_augmentation=False):
    """Parses the Google Landmark Dataset into (image, sparse_label) format.
    """
    # Create a description of the features.
    feature_description = {
        'image/height': FixedLenFeature([], tf.int64, default_value=0),
        'image/width': FixedLenFeature([], tf.int64, default_value=0),
        'image/channels': FixedLenFeature([], tf.int64, default_value=0),
        'image/format': FixedLenFeature([], tf.string, default_value=''),
        'image/id': FixedLenFeature([], tf.string, default_value=''),
        'image/filename': FixedLenFeature([], tf.string, default_value=''),
        'image/encoded': FixedLenFeature([], tf.string, default_value=''),
        'image/class/label': FixedLenFeature([], tf.int64, default_value=0),
    }
    customized_parse_func = functools.partial(
          _parse_function,
          name_to_features=feature_description,
          image_size=image_size,
          augmentation=imagenet_augmentation
    )
    return customized_parse_func


def retrieve_tfrecord_filenames(tfrecord_dir, basename, shards):
    """Generates filenames of sharded TFRecord dataset"""
    filenames = []
    for shard_index in range(shards):
        shard_filename = os.path.join(
            tfrecord_dir, f'{basename}-{shard_index:05d}-of-{shards:05d}')
        filenames.append(shard_filename)
    return filenames

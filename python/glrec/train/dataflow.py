import os
import math
import tensorflow as tf
import functools
from tensorflow.io import FixedLenFeature
import tensorflow.keras.backend as K


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


class TFImageTransform:
    """
    Performs random image transformations on GPU or TPU. This should be
    used ONLY when your CPU speed is so slow that it becomes a bottleneck
    to your training loop. Supports ONLY SQUARE IMAGES!

    Supported transformations, divided into 2 groups by performance:
      * Fast Transformations:
        - Random horizontal/vertical flip
        - Random brightness adjustment
      * Slow Transformations:
        - Random Rotation
        - Random shear
        - Random zooming (in or out)
        - Random shifting

    Sample usage with tf.data.Dataset objects, for a classification
    dataset with images and labels:

      transformer = ImageTransform(
          image_size, rotation_range=15., shear_range=5.,
          zoom_range=(0.1, 0.1), shift_range=(16., 16.))
      dataset = dataset.map(
          lambda image, label: transformer.transform(image), label)

    # Arguments:
        image_size: a tuple (height: int, width: int)
        rotation_range: a float, degrees of left/right rotations
        shear_range: a float, degrees of left/right shear
        shift_range: a tuple (height_shift: float, width_shift: float)
        zoom_range: a tuple (height_zoom: float, width_zoom: float)
    """

    def __init__(self,
                 image_size,
                 horizontal_flip=True,
                 vertical_flip=True,
                 brightness_adjustments=True,
                 rotation_range=0.,
                 shear_range=0.,
                 shift_range=(0., 0.),
                 zoom_range=(0., 0.)):

        self.image_size = image_size

        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.brightness_adjustments = brightness_adjustments

        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range

        self.advanced_agumentations = \
            rotation_range != 0. or \
            shear_range != 0. or \
            shift_range != (0., 0.) or \
            zoom_range != (0., 0.)

    @staticmethod
    def get_transformation_matrix(rotation, shear, zoom, shift):
        """
        Generates a transformation matrix under given parameters.
        The matrix is calculated as dot product M = T * Z * S * R,
        where T is shift (translation) matrix, S is shear matrix,
        Z is zoom matrix and R is translation matrix. This matrix
        can then be applied to image A as M * A.

        # Arguments:
            rotation: in degrees, from -180.0 to 180.0
            shear: in degrees, from -180.0 to 180.0
            zoom: a tuple of floats (height zoom, width zoom)
            shift: a tuple of floats (height shift, width shift)

        # Returns: a transformation matrix
        """

        height_zoom, width_zoom = zoom
        height_shift, width_shift = shift
        one = tf.constant([1], dtype='float32')
        zero = tf.constant([0], dtype='float32')

        # convert degress to radians
        rotation = math.pi * rotation / 180.
        shear = math.pi * shear / 180.

        # calculate 2d rotation matrix
        c1 = tf.math.cos(rotation)
        s1 = tf.math.sin(rotation)
        rotation_matrix = tf.reshape(tf.concat([
            c1,   s1,   zero,
            -s1,  c1,   zero,
            zero, zero, one
        ], axis=0), [3, 3])

        # calculate shear matrix
        c2 = tf.math.cos(shear)
        s2 = tf.math.sin(shear)
        shear_matrix = tf.reshape(tf.concat([
            one,  s2,   zero,
            zero, c2,   zero,
            zero, zero, one
        ], axis=0), [3, 3])

        # calculate zoom matrix
        zoom_matrix = tf.reshape(tf.concat([
            one/height_zoom, zero,           zero,
            zero,            one/width_zoom, zero,
            zero,            zero,           one
        ], axis=0), [3, 3])

        # calculate shift matrix
        shift_matrix = tf.reshape(tf.concat([
            one,  zero, height_shift,
            zero, one,  width_shift,
            zero, zero, one
        ], axis=0), [3, 3])

        return K.dot(
            K.dot(rotation_matrix, shear_matrix),
            K.dot(zoom_matrix, shift_matrix))

    def transform(self, image):
        """
        Applies random transformations to the given image. Supported 2D image
        transformations, divided into 2 groups by performance:
          * Fast Transformations:
            - Random horizontal/vertical flip
            - Random brightness adjustment
          * Slow Transformations:
            - Random Rotation
            - Random shear
            - Random zooming (in or out)
            - Random shifting

        # Arguments:
            image: a tf.Tensor image of shape [h, w, 3]
            label: image label

        # Returns:
            image: randomly rotated, sheared, zoomed, and shifted
            label: the same as input label
        """
        # first, perform random flips and color corrections
        if self.horizontal_flip:
            image = tf.image.random_flip_left_right(image)

        if self.vertical_flip:
            image = tf.image.random_flip_up_down(image)

        if self.brightness_adjustments:
            image = tf.image.random_brightness(image, max_delta=0.2)

        # Don't calculate transformation matrices if not necessary
        if not self.advanced_agumentations:
            return image

        # The calculations below will only be performed if advanced
        # augmentations are set. So, performance-wise we're good.

        dim = self.image_size[0]
        xdim = dim % 2  # fix for odd sizes, e.g. 331

        rot = self.rotation_range * tf.random.normal([1], dtype='float32')
        shr = self.shear_range * tf.random.normal([1], dtype='float32')
        h_zoom = 1.0 + tf.random.normal(
                [1], dtype='float32') * self.zoom_range[0]
        w_zoom = 1.0 + tf.random.normal(
                [1], dtype='float32') * self.zoom_range[1]
        h_shift = self.shift_range[0] * tf.random.normal([1], dtype='float32')
        w_shift = self.shift_range[1] * tf.random.normal([1], dtype='float32')

        # calculate transformation matrix
        m = type(self).get_transformation_matrix(
            rot, shr, (h_zoom, w_zoom), (h_shift, w_shift))

        # destination pixel indices
        x = tf.repeat(tf.range(dim // 2, -dim // 2, -1), dim)
        y = tf.tile(tf.range(-dim // 2, dim // 2), [dim])
        z = tf.ones([dim * dim], dtype='int32')
        idx = tf.stack([x, y, z])

        # rotate destination pixels onto original pixels
        idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
        idx2 = K.cast(idx2, dtype='int32')
        idx2 = K.clip(idx2, -dim // 2 + xdim + 1, dim // 2)

        # find origin pixel values
        idx3 = tf.stack([dim // 2 - idx2[0, ], dim // 2 - 1 + idx2[1, ]])
        d = tf.gather_nd(image, tf.transpose(idx3))

        return tf.reshape(d, [dim, dim, 3])

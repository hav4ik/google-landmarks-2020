from abc import ABC, abstractmethod
import math
import tensorflow as tf
import tensorflow.keras.backend as K


class BaseAugmentation(ABC):
    """Base class for image augmentation objects"""
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, image):
        pass


class TFImageTransform(BaseAugmentation):
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
                 horizontal_flip=True,
                 vertical_flip=True,
                 brightness_adjustments=True,
                 rotation_range=0.,
                 shear_range=0.,
                 shift_range=(0., 0.),
                 zoom_range=(0., 0.)):

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

        h, w = tf.shape(image)[0], tf.shape(image)[1]
        assert h == w  # for rotations, we only have square matrices
        dim = h
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

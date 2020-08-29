import math
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils


class ArcMarginProduct(Layer):
    """
    Implements large margin arc distance.

    Reference:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
            blob/master/src/modeling/metric_learning.py
    """
    def __init__(self,
                 num_classes,
                 s=30,
                 m=0.50,
                 easy_margin=False,
                 ls_eps=0.0,
                 **kwargs):

        super(ArcMarginProduct, self).__init__(**kwargs)
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m

    def build(self, input_shape):
        embedding_shape, label_shape = input_shape
        self.W = self.add_weight(name='W',
                                 shape=(embedding_shape[-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 dtype='float32',
                                 trainable=True,
                                 regularizer=None)

    def call(self, inputs, training=None):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )

        training = _resolve_training(self, training)
        if not training:
            # We don't have labels if we're not in training mode
            return cosine
        else:
            sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = tf.where(cosine > 0, phi, cosine)
            else:
                phi = tf.where(cosine > self.th, phi, cosine - self.mm)
            one_hot = tf.cast(
                tf.one_hot(y, depth=self.n_classes),
                dtype=cosine.dtype
            )
            if self.ls_eps > 0:
                one_hot = (1 - self.ls_eps) * one_hot + \
                        self.ls_eps / self.n_classes

            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
            return output


class ArcFace(Layer):
    """
    Implementation of ArcFace layer, devised from the PyTorch implementation.

    Reference:
      Original Paper: https://arxiv.org/abs/1801.07698
      PyTorch implementation:
        https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
    """
    def __init__(self, num_classes, s=30.0, m=0.5, regulazer=None, **kwargs):
        super().__init__(**kwargs)
        self._n_classes = num_classes
        self._s = s
        self._m = m
        self._regularizer = regulazer

    def build(self, input_shape):
        embedding_shape, label_shape = input_shape
        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self._regularizer)

    def call(self, inputs, training=None):
        embedding, label = inputs

        # Normalize features and weights
        x = tf.nn.l2_normalize(embedding)
        w = tf.nn.l2_normalize(self._w)

        # Dot product
        logits = tf.matmul(x, w)

        training = _resolve_training(self, training)
        if not training:
            # We don't have labels if we're not in training mode
            return logits
        else:
            # Add margin
            theta = tf.math.acos(
                    K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
            target_logits = tf.math.cos(theta + self._m)
            one_hot = tf.one_hot(label, depth=self._n_classes)
            output = logits * (1. - one_hot) + target_logits * one_hot

            # Feature re-scale
            output *= self._s
            return output


class AdaCos(Layer):
    def __init__(self, num_classes, m=0.5, regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self._n_classes = num_classes
        self._s = math.sqrt(2) * math.log(num_classes - 1)
        self._m = m
        self._regularizer = regularizer

    def build(self, input_shape):
        embedding_shape, label = input_shape

    def call(self, inputs):
        pass


def _resolve_training(layer, training):
    if training is None:
        training = K.learning_phase()
    if isinstance(training, int):
        training = bool(training)
    if not layer.trainable:
        # When the layer is not trainable, it overrides the value passed from
        # model.
        training = False
    return tf_utils.constant_value(training)

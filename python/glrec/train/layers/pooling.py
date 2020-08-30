import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant


class GeneralizedMeanPooling2D(Layer):
    """
    GeM pooling, as described in https://arxiv.org/abs/1711.02512.
    This layer assumes that ReLU is already applied to the inputs.
    """
    def __init__(self, p=3., train_p=False, **kwargs):
        super().__init__(**kwargs)
        self._init_p = p
        self._train_p = train_p

    def build(self, input_shape):
        self._p = self.add_weight(name='p',
                                  shape=(),
                                  initializer=Constant(self._init_p),
                                  trainable=self._train_p)

    def call(self, inputs):
        x = tf.pow(inputs, self._p)
        x = tf.reduce_mean(x, axis=[1, 2], keepdims=False)
        x = tf.pow(x, 1. / self._p)
        return x

import math
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant
from tensorflow.python.keras.utils import tf_utils
from glrec.train.utils import resolve_training_flag


class ArcMarginProduct(Layer):
    """
    Implements large margin arc distance. This implementation was taken from
    Kaggle (https://www.kaggle.com/akensert/glrec-resnet50-arcface-tf2-2).

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

        super().__init__(**kwargs)
        self._n_classes = num_classes
        self._s = float(s)
        self._m = float(m)
        self._ls_eps = ls_eps
        self._easy_margin = easy_margin
        self._cos_m = tf.math.cos(m)
        self._sin_m = tf.math.sin(m)
        self._th = tf.math.cos(math.pi - m)
        self._mm = tf.math.sin(math.pi - m) * m

    def build(self, input_shape):
        embedding_shape, label_shape = input_shape
        self._W = self.add_weight(name='W',
                                  shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  dtype='float32',
                                  trainable=True,
                                  regularizer=None)

    def call(self, inputs, training=None):
        X, y = inputs

        # Squeezing is necessary for Keras. It expands the dimension to (n, 1)
        y = tf.reshape(y, [-1])

        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self._W, axis=0)
        )

        training = resolve_training_flag(self, training)
        if not training:
            # We don't have labels if we're not in training mode
            return cosine
        else:
            sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
            phi = cosine * self._cos_m - sine * self._sin_m
            if self._easy_margin:
                phi = tf.where(cosine > 0, phi, cosine)
            else:
                phi = tf.where(cosine > self._th, phi, cosine - self._mm)
            one_hot = tf.cast(
                tf.one_hot(y, depth=self._n_classes),
                dtype=cosine.dtype
            )
            if self._ls_eps > 0:
                one_hot = (1 - self._ls_eps) * one_hot + \
                        self._ls_eps / self._n_classes

            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self._s
            return output


class ArcFace(Layer):
    """
    Implementation of ArcFace layer, devised from the PyTorch implementation.

    Reference:
      Original Paper: https://arxiv.org/abs/1801.07698
      PyTorch implementation:
        https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
    """
    def __init__(self,
                 num_classes,
                 s=30.0,
                 m=0.5,
                 regulazer=None,
                 **kwargs):

        super().__init__(**kwargs)
        self._n_classes = num_classes
        self._s = float(s)
        self._m = float(m)
        self._regularizer = regulazer

    def build(self, input_shape):
        embedding_shape, label_shape = input_shape
        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self._regularizer,
                                  name='cosine_weights')

    def call(self, inputs, training=None):
        embedding, label = inputs

        # Squeezing is necessary for Keras. It expands the dimension to (n, 1)
        label = tf.reshape(label, [-1], name='label_shape_correction')

        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(embedding, axis=1, name='normalize_prelogits')
        w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights')
        cosine_sim = tf.matmul(x, w, name='cosine_similarity')

        training = resolve_training_flag(self, training)
        if not training:
            # We don't have labels if we're not in training mode
            return self._s * cosine_sim
        else:
            one_hot_labels = tf.one_hot(label,
                                        depth=self._n_classes,
                                        name='one_hot_labels')
            theta = tf.math.acos(K.clip(
                    cosine_sim, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
            selected_labels = tf.where(tf.greater(theta, math.pi - self._m),
                                       tf.zeros_like(one_hot_labels),
                                       one_hot_labels,
                                       name='selected_labels')
            final_theta = tf.where(tf.cast(selected_labels, dtype=tf.bool),
                                   theta + self._m,
                                   theta,
                                   name='final_theta')
            output = tf.math.cos(final_theta, name='cosine_sim_with_margin')
            return self._s * output


class AdaCos(Layer):
    """
    Implementation of AdaCos layer.

    References:
      Original Paper: https://arxiv.org/abs/1905.00292
      PyTorch Implementation:
        https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
        Note that this implementation has flaws (at least I think so)
      TensorFlow 1.x Implementation:
        https://github.com/taekwan-lee/adacos-tensorflow/blob/master/adacos.py
      Keras Implementation with TensorFlow 1.x backend:
        https://github.com/DaisukeIshibe/Keras-Adacos/
    """
    def __init__(self,
                 num_classes,
                 is_dynamic=True,
                 regularizer=None,
                 name='adacos',
                 **kwargs):

        super().__init__(name=name, **kwargs)
        self._n_classes = num_classes
        self._init_s = math.sqrt(2) * math.log(num_classes - 1)
        self._is_dynamic = is_dynamic
        self._regularizer = regularizer

    def build(self, input_shape):
        embedding_shape, label_shape = input_shape
        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self._regularizer,
                                  name='adacos_weight')
        if self._is_dynamic:
            self._s = self.add_weight(shape=(),
                                      initializer=Constant(self._init_s),
                                      trainable=False,
                                      aggregation=tf.VariableAggregation.MEAN,
                                      name='adacos_scale')

    def call(self, inputs, training=None):
        embedding, label = inputs

        # Squeezing is necessary for Keras. It expands the dimension to (n, 1)
        label = tf.reshape(label, [-1])

        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(embedding, axis=1)
        w = tf.nn.l2_normalize(self._w, axis=0)
        logits = tf.matmul(x, w)

        # Fixed AdaCos
        is_dynamic = tf_utils.constant_value(self._is_dynamic)
        if not is_dynamic:
            # _s is not created since we are not in dynamic mode
            output = tf.multiply(self._init_s, logits)
            return output

        training = resolve_training_flag(self, training)
        if not training:
            # We don't have labels to update _s if we're not in training mode
            return self._s * logits
        else:
            theta = tf.math.acos(
                    K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
            one_hot = tf.one_hot(label, depth=self._n_classes)
            b_avg = tf.where(one_hot < 1.0,
                             tf.exp(self._s * logits),
                             tf.zeros_like(logits))
            b_avg = tf.reduce_mean(tf.reduce_sum(b_avg, axis=1))
            theta_class = tf.gather_nd(
                    theta,
                    tf.stack([
                        tf.range(tf.shape(label)[0]),
                        tf.cast(label, tf.int32)
                    ], axis=1))
            mid_index = tf.shape(theta_class)[0] // 2 + 1
            theta_med = tf.nn.top_k(theta_class, mid_index).values[-1]

            # Since _s is not trainable, this assignment is safe. Also,
            # tf.function ensures that this will run in the right order.
            self._s.assign(
                    tf.math.log(b_avg) /
                    tf.math.cos(tf.minimum(math.pi/4, theta_med)))

            # Return scaled logits
            return self._s * logits


class CosFace(Layer):
    """
    Implementation of CosFace layer.

    References:
      Original paper: https://arxiv.org/abs/1801.09414
      PyTorch implementation:
        https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
      Keras for Tensorflow 1.x implementation:
        https://github.com/4uiiurz1/keras-arcface/blob/master/metrics.py
    """
    def __init__(self,
                 num_classes,
                 s=30.0,
                 m=0.35,
                 regularizer=None,
                 **kwargs):

        super().__init__(**kwargs)
        self._n_classes = num_classes
        self._s = float(s)
        self._m = float(m)
        self._regularizer = regularizer

    def build(self, input_shape):
        embedding_shape, label_shape = input_shape
        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self._regularizer)

    def call(self, inputs, training=None):
        embedding, label = inputs

        # Squeezing is necessary for Keras. It expands the dimension to (n, 1)
        label = tf.reshape(label, [-1], name='label_shape_correction')

        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(embedding, axis=1, name='normalize_prelogits')
        w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights')
        cosine_sim = tf.matmul(x, w, name='cosine_similarity')

        training = resolve_training_flag(self, training)
        if not training:
            # We don't have labels if we're not in training mode
            return self._s * cosine_sim
        else:
            one_hot_labels = tf.one_hot(label,
                                        depth=self._n_classes,
                                        name='one_hot_labels')
            theta = tf.math.acos(K.clip(
                    cosine_sim, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
            final_theta = tf.where(tf.cast(one_hot_labels, dtype=tf.bool),
                                   tf.math.cos(theta) - self._m,
                                   tf.math.cos(theta),
                                   name='final_theta')
            output = tf.math.cos(final_theta, name='cosine_sim_with_margin')
            return self._s * output


class CosineSimilarity(Layer):
    """Implementation of the simple CosineSimilarity layer, used by Keetar.
    """
    def __init__(self,
                 num_classes,
                 s=30.0,
                 regularizer=None,
                 **kwargs):

        super().__init__(**kwargs)
        self._n_classes = num_classes
        self._s = float(s)
        self._regularizer = regularizer

    def build(self, input_shape):
        embedding_shape, label_shape = input_shape
        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self._regularizer)

    def call(self, inputs, training=None):
        embedding, label = inputs

        # Squeezing is necessary for Keras. It expands the dimension to (n, 1)
        label = tf.reshape(label, [-1], name='label_shape_correction')

        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(embedding, axis=1, name='normalize_prelogits')
        w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights')
        cosine_sim = tf.matmul(x, w, name='cosine_similarity')
        return self._s * cosine_sim

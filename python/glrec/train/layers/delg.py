import tensorflow as tf
import tensorflow.keras.layers as keras_layers
import tensorflow.keras.regularizers as keras_regularizers
import tensorflow.keras.backend as K
from glrec.train.utils import resolve_training_flag


class Attention(keras_layers.Layer):
    """Instantiates attention layer.

    Uses two [kernel_size x kernel_size] convolutions and softplus as
    activation to compute an attention map with the same resolution as the
    featuremap.
    """
    def __init__(self,
                 hidden_kernel_size=1,
                 hidden_num_filters=512,
                 attn_kernel_size=1,
                 regularization_decay=1e-4,
                 name='delg_attention',
                 **kwargs):
        """Initialization of attention model.

        Args:
          hidden_kernel_size: int, kernel size of hidden convolutions.
          hidden_num_filtere: int, number of filters in hidden convolutions.
          attn_kernel_size: int, kernel size of the output convolutions.
          regularization_decay: float, decay for l2 regularization of
                                kernel weights.
          name: str, name to identify model.
        """
        super().__init__(name=name, **kwargs)
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

        # First convolutional layer (called with ReLU activation)
        self.conv1 = keras_layers.Conv2D(
                filters=hidden_num_filters,
                kernel_size=hidden_kernel_size,
                kernel_regularizer=keras_regularizers.l2(regularization_decay),
                padding='same',
                name='attn_conv1')
        self.bn_conv1 = keras_layers.BatchNormalization(
                axis=bn_axis,
                name='bn_conv1')

        # Second convolution layer, with softplus activation
        self.conv2 = keras_layers.Conv2D(
                filters=1,
                kernel_size=attn_kernel_size,
                kernel_regularizer=keras_regularizers.l2(regularization_decay),
                padding='same',
                name='attn_conv2')
        self.activation_layer = keras_layers.Activation('softplus')

    def call(self, inputs, training=None):
        training = resolve_training_flag(self, training)

        x = self.conv1(inputs)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)

        score = self.conv2(x)
        probability = self.activation_layer(score)
        return probability, score


class Autoencoder(keras_layers.Layer):
    """Instantiates the autoencoder as in DELG paper.

    There are some nuances on the implementation of the autoencoder. See the
    discussion: https://github.com/tensorflow/models/issues/9189.
    """
    def __init__(self,
                 dim_embedding=128,
                 regularization_decay=1e-4,
                 activation='relu',
                 name='delg_autoencoder',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim_embedding = dim_embedding
        self.reg_decay = regularization_decay
        self.activation = activation

    def build(self, input_shape):
        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

        # Num channels will be the same as output channels
        num_channels = input_shape[bn_axis]

        # Linear encoder convolution
        self.encoder = keras_layers.Conv2D(
                filters=self.dim_embedding,
                kernel_size=1,
                kernel_regularizer=keras_regularizers.l2(self.reg_decay),
                padding='same',
                name='autoenc_conv1')

        # Linear decoder convolution
        self.decoder = keras_layers.Conv2D(
                filters=num_channels,
                kernel_size=1,
                kernel_regularizer=keras_regularizers.l2(self.reg_decay),
                padding='same',
                name='autoenc_conv2')

    def call(self, inputs, training=None):
        training = resolve_training_flag(self, training)

        # Compression to dim_embedding
        embedding = self.encoder(inputs)

        # Expansion back to the shape of inputs
        reconstruction = self.decoder(embedding)

        # Encoder's activation. Note that this should match the activation
        # of the backbone model (so, for EfficientNets, use swish)
        if self.activation == 'relu':
            reconstruction = tf.nn.relu(reconstruction)
        elif self.activation == 'swish':
            reconstruction = tf.nn.swish(reconstruction)

        return embedding, reconstruction

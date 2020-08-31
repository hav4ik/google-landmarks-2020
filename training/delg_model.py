import tensorflow as tf
import tensorflow.keras.applications as keras_applications
import efficientnet.tfkeras as efficientnet
from tensorflow.keras import layers as keras_layers

from glrec.train import utils as train_utils
from glrec.train.constants import constants as train_constants
from glrec.train.layers import heads as retrieval_heads
from glrec.train.layers import pooling as retrieval_pooling


# Mapping for backbone architecture modules
_backbone_architecture_module = {

    # Lightweight architectures for local testing
    'MobileNetV2': keras_applications.MobileNetV2,

    # ResNet family
    'ResNet50': keras_applications.ResNet50,
    'ResNet101': keras_applications.ResNet101,
    'ResNet152': keras_applications.ResNet152,
    'ResNet50V2': keras_applications.ResNet50V2,
    'ResNet101V2': keras_applications.ResNet101V2,
    'ResNet152V2': keras_applications.ResNet152V2,

    # DenseNet family
    'DenseNet121': keras_applications.DenseNet121,
    'DenseNet169': keras_applications.DenseNet169,
    'DenseNet201': keras_applications.DenseNet201,

    # EfficientNet family
    'EfficientNetB5': efficientnet.EfficientNetB5,
    'EfficientNetB6': efficientnet.EfficientNetB6,
    'EfficientNetB7': efficientnet.EfficientNetB7,
}


def load_backbone_model(architecture, weights, trainable):
    network_module = _backbone_architecture_module[architecture]
    weights_file = None
    if weights not in [None, 'imagenet', 'noisy-student']:
        weights_file, weights = weights, None

    backbone = network_module(include_top=False,
                              weights=weights)

    if weights_file is not None:
        # `by_name` flag is used if we're loading from a different
        # architecture, with some layers in common.
        weights_file = train_utils.resolve_file_path(weights_file)
        backbone.load_weights(weights_file, by_name=True)

    backbone.trainable = trainable
    return backbone


def load_global_head(layer, kwargs):
    if not hasattr(retrieval_heads, layer):
        raise ValueError(
                f'Module `glrec.layers.heads` does not contain {layer}')
    head_layer = getattr(retrieval_heads, layer)(
            num_classes=train_constants.NUM_CLASSES, **kwargs)
    return head_layer


# Mapping for pooling layers in the retrieval branch
_retrieval_pooling_module = {
    'GAP': tf.keras.layers.GlobalAveragePooling2D,
    'GeM': retrieval_pooling.GeneralizedMeanPooling2D,
}


def load_pooling_layer(method, kwargs):
    pooling_layer_class = _retrieval_pooling_module[method]
    pooling_layer = pooling_layer_class(**kwargs)
    return pooling_layer


class DelgGlobalBranch(tf.keras.layers.Layer):
    """
    Global (retrieval) branch with Cosine head. The Cosine head requires
    ground-truth to calculate margin and scale values. However, during
    inference (with K.learning_phase() == False), one can just put empty
    labels in, as it doesn't affect the outcome.
    """
    def __init__(self,
                 pooling_config,
                 embedding_dim,
                 head_config,
                 **kwargs):

        super().__init__(**kwargs)
        self._pool_features = load_pooling_layer(**pooling_config)
        self._reduce_dimensionality = keras_layers.Dense(embedding_dim)
        self._cosine_head = load_global_head(**head_config)
        self._softmax = tf.keras.layers.Softmax()

    def call(self, inputs):
        backbone_features, labels = inputs
        pooled_features = self._pool_features(backbone_features)
        dim_reduced_features = self._reduce_dimensionality(pooled_features)
        output_logits = self._cosine_head([dim_reduced_features, labels])
        output = self._softmax(output_logits)
        return output


class DelgModel(tf.keras.Model):
    """DELG architecture, as in https://arxiv.org/abs/2001.05027
    """
    def __init__(self,
                 backbone_config,
                 global_branch_config,
                 local_branch_config,
                 **kwargs):

        super().__init__(**kwargs)
        self.backbone = load_backbone_model(**backbone_config)
        self.global_branch = DelgGlobalBranch(**global_branch_config)

    def call(self, inputs):
        input_image, sparse_label = inputs
        backbone_features = self.backbone(input_image)
        global_output = self.global_branch([backbone_features, sparse_label])
        return global_output

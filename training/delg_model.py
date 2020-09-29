import tensorflow as tf
import tensorflow.keras.applications as keras_applications
import efficientnet.tfkeras as efficientnet
from tensorflow.keras import layers as keras_layers
from tensorflow.keras import backend as K

from glrec.train import utils as train_utils
from glrec.train.constants import constants as train_constants
from glrec.train.layers import heads as retrieval_heads
from glrec.train.layers import pooling as retrieval_pooling
from glrec.train.layers import delg as delg_layers


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


def load_backbone_model(architecture, weights, trainable=True):
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
                 trainable=True,
                 **kwargs):

        super().__init__(**kwargs)
        self.trainable = trainable

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

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None, None],
                      dtype=tf.float32,
                      name='delg_global_infer_input'),
    ])
    def delg_inference(self, backbone_features):
        """Returns normalized embeddings, given backbone features.
        """
        pooled_features = self._pool_features(backbone_features)
        embeddings = self._reduce_dimensionality(pooled_features)
        normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        return normalized_embeddings

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None],
                      dtype=tf.float32,
                      name='delg_normalized_embeddings_input'),
    ])
    def classify_from_embedding(self, normalized_embeddings):
        """
        Given the normalized embeddings (from `delg_inference`), perform
        classification and return a tensor (batch_size, num_classes).
        """
        normalized_w = tf.nn.l2_normalize(self._cosine_head._w, axis=0)
        cosine_similarity = tf.matmul(normalized_embeddings, normalized_w)
        return cosine_similarity


class DelgLocalBranch(tf.keras.layers.Layer):
    """Local (recognition) branch with reconstruction and attention heads.
    """
    def __init__(self,
                 attention_config,
                 autoencoder_config,
                 trainable=False,
                 name='local_branch',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.trainable = trainable

        self.attention = delg_layers.Attention(**attention_config)
        self.autoencoder = delg_layers.Autoencoder(**autoencoder_config)
        self.attention_classifier = keras_layers.Dense(
                train_constants.NUM_CLASSES,
                activation='softmax', name='attention_fc')

    def call(self, inputs):
        backbone_features, labels = inputs

        # Attention and AutoEncoder
        probability, score = self.attention(backbone_features)
        embedding, reconstruction = self.autoencoder(backbone_features)

        # Classification using attention and reconstructed features
        with tf.name_scope('local_classification'):
            # WTF? There shouldn't be an l2 here!!! This is absurd!
            # features = tf.nn.l2_normalize(reconstruction, axis=-1)
            features = reconstruction
            features = tf.reduce_sum(
                    tf.multiply(features, probability),
                    [1, 2], keepdims=False)
            tf.debugging.assert_rank(
                    features, 2, message='features should have rank 2')
            classification_output = self.attention_classifier(features)

        # I'm too lazy to do this shit properly so I'll calculate the
        # reconstruction loss right here. Pls don't judge me :(
        with tf.name_scope('local_reconstruction_score'):
            cn_axis = 3 if K.image_data_format() == 'channels_last' else 1
            pointwise_l2_norm = tf.norm(
                    reconstruction - backbone_features,
                    keepdims=False, axis=-1)
            tf.debugging.assert_rank(
                    pointwise_l2_norm, 3,
                    message='pointwise_l2_norm should have rank 3')
            reconstruction_score = tf.reduce_mean(
                    tf.math.square(pointwise_l2_norm),
                    axis=[1, 2], keepdims=False)
            reconstruction_score = tf.divide(
                    reconstruction_score,
                    tf.cast(tf.shape(backbone_features)[cn_axis], tf.float32))
            tf.debugging.assert_rank(
                    reconstruction_score, 1,
                    message='reconstruction_score should have rank 1')

        # Output the classification results and reconstruction l2 loss
        return classification_output, reconstruction_score

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None, None],
                      dtype=tf.float32,
                      name='delg_local_infer_input'),
    ])
    def delg_inference(self, backbone_features):
        """Local model (descriptors) inference, given backbone features.
        """
        # Attention scores
        probability, score = self.attention(backbone_features)

        # Dimensionality reduced embeddings
        embedding = self.autoencoder.encoder(backbone_features)
        embedding = tf.nn.l2_normalize(embedding, axis=-1)

        # Output shapes: [bs, h, w, c], [bs, h, w, 1], [bs, h, w, 1]
        return embedding, probability, score


class DelgModel(tf.keras.Model):
    """DELG architecture, as in https://arxiv.org/abs/2001.05027
    """
    def __init__(self,
                 backbone_config,
                 global_branch_config,
                 local_branch_config,
                 places_branch_config,
                 shallow_layer_name,
                 training_mode,
                 inference_mode,
                 **kwargs):
        """Initialization of the DELG  model

        Args:
          backbone_config: a dictionalry of kwargs for backbone
          global_branch_config: a dict of kwargs for DelgGlobalBranch
          local_branch_config: a dict of kwarsg for DelgLocalBranch or None
          places_branch_config: not usable right now
          shallow_layer_name: name of the shallower layer to get features
        """
        super().__init__(**kwargs)
        self.training_mode = training_mode
        self.inference_mode = inference_mode

        # Prepare backbone inference with intermediate outputs
        self.backbone = load_backbone_model(**backbone_config)
        deep_features = self.backbone.layers[-1].output
        shallow_features = self.backbone.get_layer(shallow_layer_name).output
        self.backbone_infer = tf.keras.Model(
                self.backbone.input,
                outputs=[deep_features, shallow_features])

        # Construct the global branch
        if training_mode in ['global_only', 'local_and_global']:
            self.global_branch = DelgGlobalBranch(**global_branch_config)

        # Construct the local branch
        if training_mode in ['local_only', 'local_and_global']:
            self.local_branch = DelgLocalBranch(**local_branch_config)

        # If we're only training the local branch, no need to train backbone
        if training_mode == 'local_only':
            self.backbone.trainable = False

    def call(self, inputs, first_time_warmup=False):
        """
        first_time_warmup is deprecated
        """
        input_image, sparse_label = inputs
        deep_features, shallow_features = self.backbone_infer(input_image)

        # global branch
        if self.training_mode in ['global_only', 'local_and_global']:
            global_output = self.global_branch([deep_features, sparse_label])

        # local branch with stop gradients, as described in the paper
        if self.training_mode in ['local_only', 'local_and_global']:
            shallow_features = tf.identity(shallow_features)
            shallow_features = tf.stop_gradient(shallow_features)
            local_cls_output, local_recon_score = self.local_branch(
                    [shallow_features, sparse_label])

        # 3 heads for 3 losses
        if self.training_mode == 'global_only':
            return global_output
        elif self.training_mode == 'local_only':
            return local_cls_output, local_recon_score
        elif self.training_mode == 'local_and_global':
            return global_output, local_cls_output, local_recon_score
        else:
            raise RuntimeError('training_mode should be either global_only, '
                               'local_only, or local_and_global.')

    @tf.function(input_signature=[
        tf.TensorSpec(
            shape=[None, None, None, 3],
            dtype=tf.float32,
            name='delg_infer_input',
        )
    ])
    def delg_inference(self, input_image):
        deep_features, shallow_features = self.backbone_infer(input_image)

        if self.inference_mode in ['global_only', 'local_and_global']:
            global_descriptor = \
                self.global_branch.delg_inference(deep_features)
        if self.inference_mode in ['local_only', 'local_and_global']:
            local_descriptors, probability, scores = \
                self.local_branch.delg_inference(shallow_features)

        if self.inference_mode == 'global_only':
            return global_descriptor
        elif self.inference_mode == 'local_only':
            return local_descriptors, probability, scores
        elif self.inference_mode == 'local_and_global':
            return global_descriptor, local_descriptors, probability, scores
        else:
            raise RuntimeError('Inference_mode should be either global_only, '
                               'local_only, or local_and_global.')

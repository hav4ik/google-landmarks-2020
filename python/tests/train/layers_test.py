import pytest
import tensorflow as tf
import numpy as np

from glrec.train.layers.heads import ArcFace
from glrec.train.layers.heads import ArcMarginProduct
from glrec.train.layers.heads import AdaCos
from glrec.train.layers.heads import CosFace
from glrec.train.layers.heads import CosineSimilarity

from glrec.train.layers.pooling import GeneralizedMeanPooling2D


@pytest.mark.parametrize('training', [True, False])
def test_arcface(training):
    embeddings = tf.keras.Input(shape=(512,))
    label = tf.keras.Input(shape=(), dtype=tf.int32)
    output = ArcFace(num_classes=10)([embeddings, label])
    model = tf.keras.Model([embeddings, label], output)

    batch_size = 8
    embeddings = np.random.rand(batch_size, 512)
    labels = np.random.randint(0, 10, size=(batch_size, ))
    embeddings = tf.convert_to_tensor(embeddings)
    labels = tf.convert_to_tensor(labels)

    outputs = model(
            [embeddings, labels],
            training=training,
            ).numpy()
    assert outputs.shape == (batch_size, 10)


@pytest.mark.parametrize('training', [True, False])
def test_arcmarginproduct(training):
    embeddings = tf.keras.Input(shape=(512,))
    label = tf.keras.Input(shape=(), dtype=tf.int32)
    output = ArcMarginProduct(num_classes=10)([embeddings, label])
    model = tf.keras.Model([embeddings, label], output)

    batch_size = 8
    embeddings = np.random.rand(batch_size, 512)
    labels = np.random.randint(0, 10, size=(batch_size, ))
    embeddings = tf.convert_to_tensor(embeddings)
    labels = tf.convert_to_tensor(labels)

    outputs = model(
            [embeddings, labels],
            training=training,
            ).numpy()
    assert outputs.shape == (batch_size, 10)


@pytest.mark.parametrize('training, is_dynamic', [
    (False, False), (False, True), (True, False), (True, True)
])
def test_adacos(training, is_dynamic):
    embeddings = tf.keras.Input(shape=(512,))
    label = tf.keras.Input(shape=(), dtype=tf.int32)
    output = AdaCos(num_classes=10, is_dynamic=is_dynamic)([embeddings, label])
    model = tf.keras.Model([embeddings, label], output)

    batch_size = 8
    embeddings = np.random.rand(batch_size, 512)
    labels = np.random.randint(0, 10, size=(batch_size, ))
    embeddings = tf.convert_to_tensor(embeddings)
    labels = tf.convert_to_tensor(labels)

    outputs = model(
            [embeddings, labels],
            training=training,
            ).numpy()
    assert outputs.shape == (batch_size, 10)


@pytest.mark.parametrize('training', [True, False])
def test_cosface(training):
    embeddings = tf.keras.Input(shape=(512,))
    label = tf.keras.Input(shape=(), dtype=tf.int32)
    output = CosFace(num_classes=10)([embeddings, label])
    model = tf.keras.Model([embeddings, label], output)

    batch_size = 8
    embeddings = np.random.rand(batch_size, 512)
    labels = np.random.randint(0, 10, size=(batch_size, ))
    embeddings = tf.convert_to_tensor(embeddings)
    labels = tf.convert_to_tensor(labels)

    outputs = model(
            [embeddings, labels],
            training=training,
            ).numpy()
    assert outputs.shape == (batch_size, 10)


def test_cosinesimilarity():
    embeddings = tf.keras.Input(shape=(512,))
    label = tf.keras.Input(shape=(), dtype=tf.int32)
    output = CosineSimilarity(num_classes=10)([embeddings, label])
    model = tf.keras.Model([embeddings, label], output)

    batch_size = 8
    embeddings = np.random.rand(batch_size, 512)
    labels = np.random.randint(0, 10, size=(batch_size, ))
    embeddings = tf.convert_to_tensor(embeddings)
    labels = tf.convert_to_tensor(labels)

    outputs = model(
            [embeddings, labels],
            ).numpy()
    assert outputs.shape == (batch_size, 10)


def test_generalized_mean_pooling():
    last_layer_emb = tf.convert_to_tensor(np.random.rand(8, 6, 10, 16))
    gem = GeneralizedMeanPooling2D()
    output = gem(last_layer_emb).numpy()
    assert output.shape == (8, 16)

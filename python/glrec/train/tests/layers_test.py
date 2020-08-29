import pytest
from glrec.train.layers import ArcFace
from glrec.train.layers import ArcMarginProduct
from glrec.train.layers import AdaCos
from glrec.train.layers import CosFace

import tensorflow as tf
import numpy as np


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

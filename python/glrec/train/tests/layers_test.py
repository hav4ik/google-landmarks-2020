import pytest
from glrec.train.layers import ArcFace
from glrec.train.layers import ArcMarginProduct

import tensorflow as tf
import numpy as np


@pytest.fixture
def arcface_model():
    """Returns an ArcFace model with empty backbone"""
    embeddings = tf.keras.Input(shape=(512,))
    label = tf.keras.Input(shape=(), dtype=tf.int32)
    output = ArcFace(num_classes=10)([embeddings, label])
    model = tf.keras.Model([embeddings, label], output)
    return model


@pytest.fixture
def arcmarginproduct_model():
    """Returns an ArcMarginProduct model with empty backbone"""
    embeddings = tf.keras.Input(shape=(512,))
    label = tf.keras.Input(shape=(), dtype=tf.int32)
    output = ArcMarginProduct(num_classes=10)([embeddings, label])
    model = tf.keras.Model([embeddings, label], output)
    return model


@pytest.mark.parametrize('training', [True, False])
def test_arcface(arcface_model, training):
    batch_size = 8
    embeddings = np.random.rand(batch_size, 512)
    labels = np.random.randint(0, 10, size=(batch_size, ))
    embeddings = tf.convert_to_tensor(embeddings)
    labels = tf.convert_to_tensor(labels)
    outputs = arcface_model(
            [embeddings, labels],
            training=training,
            ).numpy()
    assert outputs.shape == (batch_size, 10)


@pytest.mark.parametrize('training', [True, False])
def test_arcmarginproduct(arcmarginproduct_model, training):
    batch_size = 8
    embeddings = np.random.rand(batch_size, 512)
    labels = np.random.randint(0, 10, size=(batch_size, ))
    embeddings = tf.convert_to_tensor(embeddings)
    labels = tf.convert_to_tensor(labels)
    outputs = arcmarginproduct_model(
            [embeddings, labels],
            training=training,
            ).numpy()
    assert outputs.shape == (batch_size, 10)

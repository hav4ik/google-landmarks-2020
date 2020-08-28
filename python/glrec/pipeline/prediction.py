import gc
import operator
import pathlib

import numpy as np
from scipy import spatial

from glrec.pipeline import const
from glrec.pipeline import utils
from glrec.pipeline.embedding_model import AbstractEmbeddingModel
from glrec.pipeline.rerank_base import AbstractRerankStrategy


def get_prediction_map(test_ids, train_ids_labels_and_scores, top_k):
    """Makes dict from test ids and ranked training ids, labels, scores.
    """
    prediction_map = dict()
    for test_index, test_id in enumerate(test_ids):
        hex_test_id = utils.to_hex(test_id)

        aggregate_scores = {}
        for _, label, score in train_ids_labels_and_scores[test_index][:top_k]:
            if label not in aggregate_scores:
                aggregate_scores[label] = 0
            aggregate_scores[label] += score

        label, score = max(
                aggregate_scores.items(), key=operator.itemgetter(1))
        prediction_map[hex_test_id] = {'score': score, 'class': label}

    return prediction_map


def get_predictions(model: AbstractEmbeddingModel,
                    rerank: AbstractRerankStrategy,
                    labelmap,
                    num_to_rerank,
                    top_k,
                    distance_func='cosine'):
    """Gets predictions using embedding similarity and local feature reranking.
    """
    train_image_paths = [
            x for x in pathlib.Path(
                const.INFER_TRAIN_IMAGE_DIR).rglob('*.jpg')]
    test_image_paths = [
            x for x in pathlib.Path(
                const.INFER_TEST_IMAGE_DIR).rglob('*.jpg')]

    test_ids, test_embeddings = \
        model.extract_global_features(test_image_paths)
    train_ids, train_embeddings = \
        model.extract_global_features(train_image_paths)
    train_ids_labels_and_scores = [None] * test_embeddings.shape[0]

    # Using (slow) for-loop, as distance matrix doesn't fit in memory.
    for test_index in range(test_embeddings.shape[0]):
        distances = spatial.distance.cdist(
                test_embeddings[np.newaxis, test_index, :], train_embeddings,
                distance_func)[0]

        partition = np.argpartition(distances, num_to_rerank)[:num_to_rerank]
        nearest = sorted([(train_ids[p], distances[p]) for p in partition],
                         key=lambda x: x[1])

        train_ids_labels_and_scores[test_index] = [
            (train_id, labelmap[utils.to_hex(train_id)], 1. - cosine_distance)
            for train_id, cosine_distance in nearest
        ]

    del test_embeddings
    del train_embeddings
    gc.collect()

    pre_verification_predictions = get_prediction_map(
            test_ids, train_ids_labels_and_scores)

    for test_index, test_id in enumerate(test_ids):
        train_ids_labels_and_scores[test_index] = \
            rerank.rescore_and_rerank(
                    test_id, train_ids_labels_and_scores[test_index])

    post_verification_predictions = get_prediction_map(
        test_ids, train_ids_labels_and_scores, top_k)

    return pre_verification_predictions, post_verification_predictions

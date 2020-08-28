from abc import ABC, abstractmethod
from glrec.pipeline.embedding_model import AbstractEmbeddingModel


class AbstractRerankStrategy:
    """Abstract interface for re-ranking strategies"""
    def __init__(self):
        pass

    @abstractmethod
    def rescore_and_rerank(model: AbstractEmbeddingModel,
                           test_image_id,
                           train_ids_labels_and_scores):
        """
        Returns rescored and sorted training images by local feature extraction
        """
        pass

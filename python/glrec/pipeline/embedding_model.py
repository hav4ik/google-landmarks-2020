from abc import ABC, abstractmethod
from glrec.pipeline import utils
import numpy as np


class AbstractEmbeddingModel(ABC):
    """Abstract interface for embedding models used in the pipeline"""
    def __init__(self):
        pass

    @abstractmethod
    def global_embedding_dim():
        """Returns the dimension of global embedding vector"""
        pass

    @abstractmethod
    def local_embedding_dim():
        """Returns the dimension of local embedding vector"""
        pass

    @abstractmethod
    def max_local_embedding_num():
        """Max number of returned local embedding vectors"""
        pass

    @abstractmethod
    def extract_global_features(image_paths):
        """Calculates global embedding vector from given list of images"""
        pass

    @abstractmethod
    def extract_local_features(image_path):
        """
        Calculates local embedding vectors for given image. This method
        is separated from `extract_local_features_from_image` with the
        intent to possibly allow embedding caching (e.g. save local
        embeddings/features while extracting global embeddings/features)
        """
        pass

    @abstractmethod
    def is_landmark(image_path):
        """Returns True if given image is a landmark"""
        pass


class BaseSimpleEmbeddingModel(AbstractEmbeddingModel):
    """
    Basically the same as `AbstractEmbeddingModel`, but with some
    utility functions
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def extract_global_features_from_image(image_tensor):
        """Calculates global embedding vector for given image"""
        pass

    @abstractmethod
    def extract_local_features_from_image(image_tensor):
        """Calculates local embedding vectors for given image"""
        pass

    def extract_global_features(self, image_paths):
        """Calculates global embedding vector from given list of images"""
        num_embeddings = len(image_paths)
        embeddings = np.empty((num_embeddings, self.global_embedding_dim()))

        for i, image_path in enumerate(image_paths):
            image_tensor = utils.load_image_tensor_from_path(image_path)
            embeddings[i, :] = \
                self.extract_global_features_from_image(image_tensor).numpy()

        return embeddings

    def extract_local_features(self, image_path):
        """Calculates local embeddings (keypoints, descriptors) from give image
        """
        image_tensor = utils.load_image_tensor_from_path(image_path)
        keypoints, descriptors = \
            self.extract_local_features_from_image(image_tensor)
        return keypoints.numpy(), descriptors.numpy()

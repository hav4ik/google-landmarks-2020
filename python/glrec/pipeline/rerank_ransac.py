import numpy as np
import pydegensac
from scipy import spatial
import copy

from glrec.pipeline import utils
from glrec.pipeline.embedding_model import AbstractEmbeddingModel
from glrec.pipeline.rerank_base import AbstractRerankStrategy


class RerankRANSAC(AbstractRerankStrategy):
    """
    Static class that contains implementation of re-ranking strategy that
    utilizes RANSAC.
    """
    def __init__(self,
                 max_inlier_score=20,
                 max_reprojection_error=4.0,
                 max_ransac_iterations=100_000,
                 homography_confidence=0.99):

        self.MAX_INLIER_SCORE = max_inlier_score
        self.MAX_REPROJECTION_ERROR = max_reprojection_error
        self.MAX_RANSAC_ITERATIONS = max_ransac_iterations
        self.HOMOGRAPHY_CONFIDENCE = homography_confidence

    def get_putative_matching_keypoints(self,
                                        test_keypoints,
                                        test_descriptors,
                                        train_keypoints,
                                        train_descriptors,
                                        max_distance=0.9):
        """
        Finds matches from `test_descriptors` to KD-tree of `train_descriptors`
        """

        train_descriptor_tree = spatial.cKDTree(train_descriptors)
        _, matches = train_descriptor_tree.query(
            test_descriptors, distance_upper_bound=max_distance)

        test_kp_count = test_keypoints.shape[0]
        train_kp_count = train_keypoints.shape[0]

        test_matching_keypoints = np.array([
            test_keypoints[i, ]
            for i in range(test_kp_count)
            if matches[i] != train_kp_count
        ])

        train_matching_keypoints = np.array([
            train_keypoints[matches[i], ]
            for i in range(test_kp_count)
            if matches[i] != train_kp_count
        ])

        return test_matching_keypoints, train_matching_keypoints

    def get_num_inliers(self,
                        test_keypoints,
                        test_descriptors,
                        train_keypoints,
                        train_descriptors):
        """
        Returns the number of RANSAC inliers.
        """
        test_match_kp, train_match_kp = \
            self.get_putative_matching_keypoints(
                    test_keypoints, test_descriptors,
                    train_keypoints, train_descriptors)

        min_kp = 4  # Min keypoints supported by `pydegensac.findHomography()`
        if test_match_kp.shape[0] <= min_kp:
            return 0

        try:
            _, mask = pydegensac.findHomography(
                    test_match_kp, train_match_kp,
                    self.MAX_REPROJECTION_ERROR,
                    self.HOMOGRAPHY_CONFIDENCE,
                    self.MAX_RANSAC_ITERATIONS)

        except np.linalg.LinAlgError:  # When det(H)=0, can't invert matrix.
            return 0

        return int(copy.deepcopy(mask).astype(np.float32).sum())

    def get_total_score(self, num_inliers, global_score):
        local_score = \
                min(num_inliers, self.MAX_INLIER_SCORE) / self.MAX_INLIER_SCORE
        return local_score + global_score

    def rescore_and_rerank(self,
                           model: AbstractEmbeddingModel,
                           test_image_id,
                           train_ids_labels_and_scores):
        """
        Returns rescored and sorted training images by local feature extraction
        """
        test_image_path = utils.get_image_path('test', test_image_id)
        test_keypoints, test_descriptors = \
            model.extract_local_features(test_image_path)

        for i in range(len(train_ids_labels_and_scores)):
            train_image_id, label, global_score = \
                train_ids_labels_and_scores[i]

            train_image_path = utils.get_image_path('train', train_image_id)
            train_keypoints, train_descriptors = model.extract_local_features(
                train_image_path)

            num_inliers = self.get_num_inliers(
                    test_keypoints, test_descriptors,
                    train_keypoints, train_descriptors)
            total_score = self.get_total_score(
                    num_inliers, global_score)
            train_ids_labels_and_scores[i] = (
                    train_image_id, label, total_score)

        train_ids_labels_and_scores.sort(key=lambda x: x[2], reverse=True)
        return train_ids_labels_and_scores

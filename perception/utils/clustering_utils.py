import numpy as np
from sklearn.cluster import MeanShift


class KeypointClustering:
    def __init__(self):
        self.mean_shift = MeanShift(bandwidth=2.0, cluster_all=True,
                max_iter=25,
                min_bin_freq=1, n_jobs=1)
        self.past_clusters = None

    def __call__(self, indices, probabilities):
        """
        indices: N x D numpy array of image coordinates to be clustered.
        probabilities: N numpy array in range 0-1, predicted probability of the
            corresponding index being an actual keypoint.
        returns: C x D keypoint estimates. C is the amount of clusters found.
        """
        if self.past_clusters is not None:
            self.mean_shift.set_params(seeds=self.past_clusters)
        self.mean_shift.fit(indices)
        C, D = self.mean_shift.cluster_centers_.shape
        estimates = np.zeros((C, D), dtype=probabilities.dtype)
        for i in range(C):
            in_cluster = self.mean_shift.labels_ == i
            cluster_indices = indices[in_cluster]
            weights = probabilities[in_cluster]
            weights /= weights.sum()
            estimates[i, :] = (weights[:, None] * cluster_indices).sum(axis=0)
        self.past_clusters = self.mean_shift.cluster_centers_
        return estimates


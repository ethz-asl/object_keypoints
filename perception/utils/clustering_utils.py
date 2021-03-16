import numpy as np
from sklearn import cluster


class KeypointClustering:
    def __init__(self, K, bandwidth):
        self.K = K
        self.clustering = cluster.KMeans(n_clusters=K, n_init=1)
        self.past_clusters = None

    def __call__(self, indices, probabilities):
        """
        indices: N x D numpy array of image coordinates to be clustered.
        probabilities: N numpy array in range 0-1, predicted probability of the
            corresponding index being an actual keypoint.
        returns: C x D keypoint estimates. C is the amount of clusters found.
        """
        D = indices.shape[1]
        if self.past_clusters is not None:
            self.clustering.set_params(init=self.past_clusters)

        if indices.shape[0] < self.K:
            # We have less points than clusters. Just return the points.
            out = np.zeros((self.K, D), dtype=indices.dtype)
            out[:indices.shape[0]] = indices
            return out

        self.clustering.fit(indices)

        estimates = self.clustering.cluster_centers_
        for i in range(estimates.shape[0]):
            in_cluster = self.clustering.labels_ == i
            cluster_indices = indices[in_cluster]
            weights = probabilities[in_cluster]
            weights /= weights.sum()
            estimates[i, :] = (weights[:, None] * cluster_indices).sum(axis=0)
        self.past_clusters = estimates
        return estimates

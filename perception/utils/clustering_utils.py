import numpy as np
from sklearn import cluster


class KeypointClustering:
    def __init__(self, K, bandwidth):
        self.K = K
        self.clustering = cluster.MeanShift(bandwidth=bandwidth, cluster_all=True)
        self.past_clusters = None

    def __call__(self, indices, probabilities):
        """
        indices: N x D numpy array of image coordinates to be clustered.
        probabilities: N numpy array in range 0-1, predicted probability of the
            corresponding index being an actual keypoint.
        returns: C x D keypoint estimates. C is the amount of clusters found.
        """
        D = indices.shape[1]
        if indices.shape[0] < self.K:
            # We have less points than clusters. Just return the points.
            out = np.zeros((self.K, D), dtype=indices.dtype)
            out[:indices.shape[0]] = indices
            return out

        self.clustering.fit(indices)

        estimates = np.zeros((self.K, D))
        centers = self.clustering.cluster_centers_
        if centers.shape[0] < self.K:
            print(f"Found only {centers.shape[0]} centers.")
        K = min(self.K, centers.shape[0])
        for i in range(K):
            in_cluster = self.clustering.labels_ == i
            cluster_indices = indices[in_cluster]
            # weights = probabilities[in_cluster]
            # weights /= weights.sum()
            # estimates[i, :] = (weights[:, None] * cluster_indices).sum(axis=0)
            estimates[i, :] = centers[i]

        self.past_clusters = estimates

        return self.past_clusters

import numpy as np
from sklearn import cluster


class KeypointClustering:
    def __init__(self, bandwidth):
        self.clustering = cluster.MeanShift(bandwidth=bandwidth, cluster_all=True, bin_seeding=True,
                min_bin_freq=1)
        self.past_clusters = None

    def __call__(self, indices):
        """
        indices: N x D numpy array of image coordinates to be clustered.
        returns: C x D keypoint estimates. C is the amount of clusters found.
        """
        self.clustering.fit(indices)
        return self.clustering.cluster_centers_


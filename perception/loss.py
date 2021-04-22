import torch
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F

class KeypointLoss(_Loss):
    def __init__(self, keypoint_config, size_average=None, reduce=None, reduction='sum'):
        super().__init__(size_average, reduce, reduction)
        self.negative_weight = 0.01
        self.positive_weight = 0.99
        self.keypoint_config = keypoint_config
        self.n_keypoint_maps = len(keypoint_config) + 1 # Add one for center map.
        if reduction == 'mean':
            self.reduce = torch.mean
        elif reduction == 'sum':
            self.reduce = torch.sum
        else:
            raise NotImplementedError("Unknown reduction method {reduction}, try 'mean' or 'sum'.")

    def forward(self, heatmap_predictions, heatmap_gt, center_predictions, center_gt):
        """
        predictions: N x D x H x W prediction tensor
        gt: N x D x H x W
        """
        nonzero_examples = heatmap_gt > 1e-3

        heatmap_loss = F.mse_loss(torch.sigmoid(heatmap_predictions), heatmap_gt, reduction='none')
        positive_loss = heatmap_loss[nonzero_examples].sum() * self.positive_weight
        negative_loss = heatmap_loss[nonzero_examples == False].sum() * self.negative_weight
        heatmap_loss = self.reduce(positive_loss + negative_loss)

        # Use only vectors from areas where there are keypoints.
        # N x 2 x H x W mask for vectors which are on top of keypoints
        N2HW_positive = heatmap_gt.sum(dim=1)[:, None].expand(-1, 2, -1, -1) > 0.25

        center_predictions = center_predictions[N2HW_positive]
        center_gt = center_gt[N2HW_positive]
        regression_loss = self.reduce(F.l1_loss(center_predictions, center_gt, reduction=self.reduction))

        return (heatmap_loss + regression_loss) / heatmap_predictions.shape[0]


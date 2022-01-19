import torch
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F

class KeypointLoss(_Loss):
    def __init__(self, keypoint_config, depth_weight=10.0, center_weight=1.0, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.keypoint_config = keypoint_config
        self.n_keypoint_maps = len(keypoint_config) + 1 # Add one for center map.
        self.depth_weight = depth_weight
        self.center_weight = center_weight
        if reduction == 'mean':
            self.reduce = torch.mean
        elif reduction == 'sum':
            self.reduce = torch.sum
        else:
            raise NotImplementedError("Unknown reduction method {reduction}, try 'mean' or 'sum'.")

    def forward(self, p_heatmaps, gt_heatmaps, p_depth, gt_depth, p_centers, gt_centers):
        """
        predictions: N x D x H x W prediction tensor
        gt: N x D x H x W
        """
        heatmap_loss = 0.0
        depth_loss = 0.0
        center_loss = 0.0
        heatmap_losses = []
        N = float(gt_heatmaps.shape[0])
        depth_losses = []
        center_losses = []
        for p_hm, p_d, p_center in zip(p_heatmaps, p_depth, p_centers):
            loss = F.binary_cross_entropy_with_logits(p_hm, gt_heatmaps, reduction='none').sum(dim=[1,2,3]).mean()
            heatmap_loss += loss
            heatmap_losses.append(loss)

            where_heat = gt_heatmaps > 0.01

            depth_l1 = F.l1_loss(p_d[where_heat], gt_depth[where_heat], reduction='sum')
            depth_loss += depth_l1 / N
            depth_losses.append(depth_l1)

            where_heat = where_heat[:, 1:, None].expand(-1, -1, 2, -1, -1)
            center_l1 = F.smooth_l1_loss(p_center[where_heat], gt_centers[where_heat], reduction='sum')
            center_loss += center_l1 / N
            center_losses.append(center_l1)

        loss = heatmap_loss + self.depth_weight * depth_loss + self.center_weight * center_loss
        return loss, heatmap_losses, depth_losses, center_losses


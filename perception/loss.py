import torch
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F
from perception.models import spatial_softmax

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

class IntegralRegression(_Loss):
    def __init__(self, keypoint_config, center_weight=1.0):
        super().__init__()
        self.keypoint_config = keypoint_config
        self.n_keypoint_maps = len(keypoint_config) + 1 # Add one for center map.
        self.center_weight = center_weight

    def _heatmap_to_normalized_coordinates(self, heatmap):
        """
        heatmap: N x K x H X W
        """
        N, K, H, W = heatmap.shape
        x = heatmap.sum(dim=2)
        y = heatmap.sum(dim=3)

        x = x * torch.arange(W, device=x.device)[None, None]
        y = y * torch.arange(H, device=y.device)[None, None]
        x = x.sum(dim=2, keepdim=True)
        y = y.sum(dim=2, keepdim=True)
        x *= 1.0 / float(W)
        y *= 1.0 / float(H)
        x -= 0.5
        y -= 0.5
        return x, y

    def _depth_integration(self, heatmap, depth):
        N, K, H, W = heatmap.shape
        depth_prediction = heatmap * depth
        return depth_prediction.reshape(N, K, -1).sum(dim=2, keepdim=True)

    def forward(self, p_heatmaps, gt_heatmaps, p_depth, gt_depth, p_centers, gt_centers):
        center_loss = 0.0
        N = float(gt_heatmaps.shape[0])
        center_losses = []
        p_heatmaps_softmax = spatial_softmax(p_heatmaps)
        coord_x, coord_y = self._heatmap_to_normalized_coordinates(p_heatmaps_softmax)
        depth_prediction = self._depth_integration(p_heatmaps_softmax, p_depth)
        xyz = torch.cat([coord_x, coord_y, depth_prediction], dim=2)

        x_gt, y_gt = self._heatmap_to_normalized_coordinates(gt_heatmaps)
        depth_gt = self._depth_integration(gt_heatmaps, gt_depth)
        xyz_gt = torch.cat([x_gt, y_gt, depth_gt], dim=2)
        integral_loss = F.l1_loss(xyz, xyz_gt)

        for p_center, gt_center, gt_hm in zip(p_centers, gt_centers, gt_heatmaps):
            where_heat = gt_hm.sum(dim=0) > 0.01

            where_heat = where_heat[None].expand(2, -1, -1)
            center_l1 = F.smooth_l1_loss(p_center[where_heat], gt_center[where_heat], reduction='sum')
            center_loss += center_l1 / N
            center_losses.append(center_l1)

        loss = integral_loss + self.center_weight * center_loss
        return loss, integral_loss, sum(center_losses)

import torch
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F

class FocalLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.gamma = 2.0
        self.alpha = 1.0

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p = torch.sigmoid(pred)
        diff = torch.abs(target - p)
        return self.alpha * diff ** self.gamma * bce


class KeypointLoss(_Loss):
    def __init__(self, keypoint_config, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.keypoint_config = keypoint_config
        self.n_keypoint_maps = len(keypoint_config) + 1 # Add one for center map.
        self.heatmap_weight = 1.0
        self.center_weight = 1.0
        self.depth_weight = 3.0
        self.focal_loss = FocalLoss()
        if reduction == 'mean':
            self.reduce = torch.mean
        elif reduction == 'sum':
            self.reduce = torch.sum
        else:
            raise NotImplementedError("Unknown reduction method {reduction}, try 'mean' or 'sum'.")

    def forward(self, heatmap_predictions, heatmap_gt, depth_p, depth_gt, center_predictions, center_gt):
        """
        predictions: N x D x H x W prediction tensor
        gt: N x D x H x W
        """
        # heatmap_loss = F.binary_cross_entropy(torch.sigmoid(heatmap_predictions), heatmap_gt, reduction='none')
        heatmap_loss = self.focal_loss(heatmap_predictions, heatmap_gt)
        heatmap_loss = heatmap_loss.sum(dim=[1, 2, 3]).mean()

        where_depth = depth_gt > 0.01
        depth_loss = self.reduce(F.l1_loss(depth_p[where_depth], depth_gt[where_depth], reduction='none'))

        # Use only vectors from areas where there are keypoints.
        # N x 2 x H x W mask for vectors which are on top of keypoints

        N, D, two, H, W = center_gt.shape
        center_predictions = center_predictions.reshape(N, D, two, H, W)

        where_keypoints = (heatmap_gt[:, 1:, None] > 0.05).expand(-1, -1, 2, -1, -1)
        regression_loss = self.reduce(F.l1_loss(center_predictions[where_keypoints], center_gt[where_keypoints], reduction='none'))

        losses = (heatmap_loss, depth_loss, regression_loss)
        return (self.heatmap_weight * heatmap_loss + self.depth_weight * depth_loss + self.center_weight * regression_loss), losses

if __name__ == "__main__":
    focal_loss = FocalLoss()
    target = torch.empty(10, 10).random_(2)
    parameter = torch.randn(10, 10, requires_grad=True)
    optimizer = torch.optim.SGD(lr=100.0, params=[parameter])
    for _ in range(10000):
        loss = focal_loss(parameter, target).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('loss: ', loss, end='\r')
    print("")
    print("diff: ", torch.abs(target - torch.sigmoid(parameter)).mean())


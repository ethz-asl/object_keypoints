import torch
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F

class FocalLoss(_Loss):
    def __init__(self):
        super().__init__()
        self.gamma = 2.0
        self.alpha = 0.25

    def forward(self, pred, target):
        p = torch.sigmoid(pred)
        bce = F.binary_cross_entropy(p, target)
        p_t = (target > 0.5) * p + (target < 0.5) * (1.0 - p)
        return self.alpha * (1.0 - p_t) ** self.gamma * bce


class KeypointLoss(_Loss):
    def __init__(self, keypoint_config, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.keypoint_config = keypoint_config
        self.n_keypoint_maps = len(keypoint_config) + 1 # Add one for center map.
        self.center_weight = 10.0
        self.focal_loss = FocalLoss()
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
        # heatmap_loss = F.binary_cross_entropy(torch.sigmoid(heatmap_predictions), heatmap_gt, reduction='none')
        heatmap_loss = self.focal_loss(heatmap_predictions, heatmap_gt)
        heatmap_loss = heatmap_loss.sum(dim=[2,3]).mean()

        # Use only vectors from areas where there are keypoints.
        # N x 2 x H x W mask for vectors which are on top of keypoints
        N2HW_positive = heatmap_gt.sum(dim=1)[:, None].expand(-1, 2, -1, -1) > 0.01

        regression_loss = self.reduce(F.l1_loss(center_predictions[N2HW_positive], center_gt[N2HW_positive], reduction='none'))

        return (heatmap_loss + self.center_weight * regression_loss)

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


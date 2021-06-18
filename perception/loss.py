import torch
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F

class FocalLoss(_Loss):
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = torch.tensor(1e-30)

    def forward(self, pred, target):
        N = target.shape[0]
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p = torch.sigmoid(pred)
        mask = (target > 0.75).float()
        return (
            mask * torch.pow(1.0 - p, self.alpha) +
            (1.0 - mask) * torch.pow(1.0 - target, self.beta) * torch.pow(p, self.alpha)
        ) * bce

class KeypointLoss(_Loss):
    def __init__(self, keypoint_config, center_weight=1.0, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.keypoint_config = keypoint_config
        self.n_keypoint_maps = len(keypoint_config) + 1 # Add one for center map.
        self.focal_loss = FocalLoss()
        self.center_weight = center_weight
        if reduction == 'mean':
            self.reduce = torch.mean
        elif reduction == 'sum':
            self.reduce = torch.sum
        else:
            raise NotImplementedError("Unknown reduction method {reduction}, try 'mean' or 'sum'.")

    def forward(self, p_heatmaps, gt_heatmaps, p_centers, gt_centers):
        """
        predictions: N x D x H x W prediction tensor
        gt: N x D x H x W
        """
        heatmap_loss = 0.0
        center_loss = 0.0
        heatmap_losses = []
        N = float(gt_heatmaps.shape[0])
        center_losses = []
        for p_hm, p_center in zip(p_heatmaps, p_centers):
            loss = self.focal_loss(p_hm, gt_heatmaps).sum(dim=[1,2,3]).mean()
            heatmap_loss += loss
            heatmap_losses.append(loss)

            where_heat = gt_heatmaps > 0.01
            where_heat = where_heat[:, 1:, None].expand(-1, -1, 2, -1, -1)
            center_l1 = F.smooth_l1_loss(p_center[where_heat], gt_centers[where_heat], reduction='sum')
            center_loss += center_l1 / N
            center_losses.append(center_l1)

        loss = heatmap_loss + self.center_weight * center_loss
        return loss, heatmap_losses, center_losses

if __name__ == "__main__":
    focal_loss = FocalLoss(1.0, 1.0)
    target = torch.clip(torch.randn(10, 10) + torch.empty(10, 10).random_(2), 0.0, 1.0)
    parameter = torch.randn(10, 10, requires_grad=True)
    optimizer = torch.optim.SGD(lr=1000.0, params=[parameter])
    for _ in range(10000):
        loss = focal_loss(parameter, target).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('loss: ', loss, end='\r')
    print("")
    print((target > 0.99).sum())
    print("positive correct: ", (torch.sigmoid(parameter)[target > 0.99] > 0.99).sum())
    print("negative correct: ", (torch.sigmoid(parameter)[target < 0.99] < 0.01).sum())
    print("diff: ", torch.abs(target - torch.sigmoid(parameter)).mean())
    print("max diff: ", torch.abs(target - torch.sigmoid(parameter)).max())


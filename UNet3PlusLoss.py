import torch
import torch.nn as nn

class UNet3PlusLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.focal_loss = FocalLoss()
        self.iou_loss = IOULoss()
        self.ms_ssim_loss = MS_SSIMLoss()

    def forward(self, y_pred, y_true):

        focal_loss = self.focal_loss(y_pred, y_true)
        iou_loss = self.iou_loss(y_pred, y_true)
        ms_ssim_loss = self.ms_ssim_loss(y_pred, y_true)

        return focal_loss + iou_loss + ms_ssim_loss

class FocalLoss(nn.Module):
    def __init__(self, weight = None, gamma = 2, reduction = 'mean'):
        super().__init__()

        self.weight = weight
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        loss = nn.functional.cross_entropy(y_pred, y_true)
        pt = torch.exp(-loss)
        focal_loss = ((1 - pt) ** self.gamma * loss)

        return focal_loss.mean()

class IOULoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.epsilon = 1e-6

    def forward(self, y_pred, y_true):

        y_pred_flat = y_pred.flatten(start_dim = 1)
        y_true_flat = y_true.flatten(start_dim = 1)

        intersection = torch.sum(y_pred_flat * y_true_flat)
        union = y_pred_flat.sum() + y_true_flat.sum() - intersection

        # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10

        return 1 - (intersection + self.epsilon) / (union + self.epsilon)

class MS_SSIMLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred, y_true, betas = [1],gammas = [1]):
        pass
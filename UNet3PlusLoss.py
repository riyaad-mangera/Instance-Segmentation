import torch
import torch.nn as nn
from torchmetrics import JaccardIndex
from torchmetrics import Dice
from ssim import MS_SSIM
from focal_loss import FocalLoss

class UNet3PlusLoss(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.weight = 0.25

        # self.cross_entropy_loss = nn.CrossEntropyLoss()
        # self.dice_loss = Dice_Loss()
        self.focal_loss = Focal_Loss()
        self.iou_loss = IOU_Loss() 
        self.ms_ssim_loss = MS_SSIM_Loss()

    def forward(self, y_pred, y_true, weight = 1):

        focal_loss = self.focal_loss(y_pred, torch.argmax(y_true, dim = 1))
        iou_loss = 1 - self.iou_loss(y_pred, y_true)
        ms_ssim_loss = 1 - self.ms_ssim_loss(y_pred, y_true)

        loss = (focal_loss + iou_loss + ms_ssim_loss) * weight #(dice_loss * self.weight) + (cross_entropy_loss * (1 - self.weight))

        return torch.sum(loss)
    
class Dice_Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.dice = Dice(num_classes = 9)

    def forward(self, y_pred, y_true):

        return self.dice(y_pred, torch.argmax(y_true, dim = 1))

class Focal_Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.focal_loss = FocalLoss()

    def forward(self, y_pred, y_true):

        focal_loss = self.focal_loss(y_pred, y_true)

        return focal_loss

class IOU_Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.iou = JaccardIndex(task = "multiclass", num_classes = 9)

    def _iou(self, y_pred, y_true):

        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        y_true = torch.argmax(y_true, dim = 1)

        iou = self.iou(y_pred, y_true)

        return iou

    def forward(self, y_pred, y_true):
        
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        y_true = torch.argmax(y_true, dim = 1)

        return self.iou(y_pred, y_true)

class MS_SSIM_Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.ms_ssim = MS_SSIM(data_range = 1, channel = 9)
        
    def _msssim(self, y_pred, y_true):

        y_pred = torch.nn.functional.softmax(y_pred, dim=1)

        ms_ssim = self.ms_ssim(y_pred, y_true)

        return ms_ssim

    def forward(self, y_pred, y_true):
        
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)

        return self.ms_ssim(y_pred, y_true)

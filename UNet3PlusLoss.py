import torch
import torch.nn as nn
import torchvision
from torchmetrics import JaccardIndex
from ssim import MS_SSIM
from math import exp

class UNet3PlusLoss(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.weight = 0.25

        # self.cross_entropy_loss = nn.CrossEntropyLoss()
        # self.dice_loss = DiceLoss()
        # self.focal_loss = FocalLoss()
        self.iou_loss = JaccardIndex(task = "multiclass", num_classes = 9, ignore_index = 0) #IOULoss()
        self.ms_ssim_loss = MS_SSIM(data_range = 1, channel = 9, K = (0.01, 0.4), weights = [0.008, 0.124, 0.124, 0.124, 0.124, 0.124, 0.124, 0.124, 0.124]) #MS_SSIMLoss(self.device)

    def forward(self, y_pred, y_true):

        # y_pred = torch.nn.Softmax(dim=1)(y_pred) #torch.nn.functional.sigmoid(y_pred)

        # cross_entropy_loss = self.cross_entropy_loss(y_pred, y_true)
        # dice_loss = self.dice_loss(y_pred, y_true)
        focal_loss = torchvision.ops.sigmoid_focal_loss(y_pred, y_true, reduction = "mean") #self.focal_loss(y_pred, y_true)
        iou_loss = self.iou_loss(torch.nn.Softmax(dim=1)(y_pred), y_true)
        ms_ssim_loss = self.ms_ssim_loss(torch.nn.Softmax(dim=1)(y_pred), y_true)

        # print("A")
        # print(focal_loss)
        # print(iou_loss)
        # print(ms_ssim_loss)

        # if focal_loss < 0:
        #     print("Focal Negative")
        # elif iou_loss < 0:
        #     print("IOU Negative")
        # elif ms_ssim_loss < 0:
        #     print("MS SSIM Negative")

        loss = focal_loss + iou_loss + ms_ssim_loss #(dice_loss * self.weight) + (cross_entropy_loss * (1 - self.weight)) #focal_loss + iou_loss #+ ms_ssim_loss

        return torch.sum(loss)

class DiceLoss(nn.Module):
    def __init__(self, epsilon = 1e-6):
        super().__init__()

        self.epsilon = epsilon

    def forward(self, y_pred, y_true):

        # y_pred = torch.nn.functional.sigmoid(y_pred)

        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)

        dice = (2.0 * (intersection + self.epsilon)) / (union + self.epsilon)

        return 1.0 - dice

class FocalLoss(nn.Module):
    def __init__(self, weight = None, gamma = 2, reduction = 'mean'):
        super().__init__()

        self.weight = weight
        self.gamma = gamma

        # self.cross_entropy_loss = nn.CrossEntropyLoss()

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

        iou = (intersection + self.epsilon) / (union + self.epsilon)

        thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10

        return 1 - iou

class MS_SSIMLoss(nn.Module):
    def __init__(self, device, window_size=11, size_average=True):
        super().__init__()

        self.device = device

        self.window_size = window_size
        self.size_average = size_average

        self.C1 = 1e-4
        self.C2 = 9e-4

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2 / float(2* sigma**2)) for x in range(window_size)])

        return gauss/gauss.sum()

    def create_window(self, window_size, channel = 1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        
        return window

    def ssim(self, y_pred, y_true, window_size=11, window=None, size_average=True, full=False, val_range=None):
        min_val = 0
        max_val = 1

        L = max_val - min_val

        b, c, h, w = y_pred.shape

        if window is None:
            real_size = min(window_size, h, w)
            window = self.create_window(real_size, channel = c).to(self.device)

        mu1 = torch.nn.functional.conv2d(y_pred, window, padding = 0, groups = c)
        mu2 = torch.nn.functional.conv2d(y_true, window, padding = 0, groups = c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.nn.functional.conv2d(y_pred * y_pred, window, padding = 0, groups = c) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(y_true * y_true, window, padding = 0, groups = c) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(y_pred * y_true, window, padding = 0, groups = c) - mu1_mu2

        self.C1 = (0.01 * L) ** 2
        self.C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + self.C2
        v2 = sigma1_sq + sigma2_sq + self.C2
        cs = v1 / v2

        ssim_map = ((2 * mu1_mu2 + self.C1) * v1) / ((mu1_sq + mu2_sq + self.C1) * v2)

        # ssim_map = (ssim_map + 1) / 2
        # cs = (cs + 1) / 2

        return ssim_map.mean(), cs.mean()

    def msssim(self, y_pred, y_true, window_size=11, size_average=True, val_range=None, normalize=None):
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(self.device)
        levels = weights.size()[0]
        ssims = []
        mcs = []

        for i in range(levels):
            sim, cs = self.ssim(y_pred, y_true, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

            ssims.append(sim)
            mcs.append(cs)

            y_pred = torch.nn.functional.avg_pool2d(y_pred, (2, 2))
            y_true = torch.nn.functional.avg_pool2d(y_true, (2, 2))

        ssims = torch.stack(ssims)
        mcs = torch.stack(mcs)

        pow1 = mcs **weights
        pow2 = ssims **weights

        return torch.prod(pow1[:-1]) * pow2[-1]

    def forward(self, y_pred, y_true):

        y_pred = torch.nn.functional.sigmoid(y_pred)

        return self.msssim(y_pred, y_true, window_size = self.window_size, size_average = self.size_average)



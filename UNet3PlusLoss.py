import torch
import torch.nn as nn
import torchvision
from torchmetrics import JaccardIndex
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.segmentation import MeanIoU
from ssim import MS_SSIM
from ssim_v2 import SSIM
from focal import FocalLossWithLogits
from focal_loss import FocalLoss
from math import exp

class UNet3PlusLoss(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.weight = 0.25

        # self.cross_entropy_loss = nn.CrossEntropyLoss()
        # self.dice_loss = DiceLoss()
        self.focal_loss = Focal_Test() #FocalLoss()
        self.iou_loss = IOU_Test() #mIoULoss(n_classes = 9) #, weight = torch.Tensor([0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]).to(self.device)) #JaccardIndex(task = "multiclass", num_classes = 9, ignore_index = 0) #IOULoss()
        self.ms_ssim_loss = MS_SSIM_Test() #MS_SSIMLoss(device = self.device) #MS_SSIM() #MS_SSIM_Test() #MS_SSIM(data_range = 1, channel = 9) #, weights = [1.0]) #, K = (0.01, 0.03), weights = [0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]) #MS_SSIMLoss(self.device)

    def forward(self, y_pred, y_true, weight = 1):

        # y_pred = torch.nn.Softmax(dim=1)(y_pred) #torch.nn.functional.sigmoid(y_pred)

        # cross_entropy_loss = self.cross_entropy_loss(y_pred, y_true)
        # dice_loss = self.dice_loss(y_pred, y_true)

        # y_pred_fc = torch.cat((y_pred[:, :0, :, :], y_pred[:, 1:, :, :]), dim=1)
        # y_true_fc = torch.cat((y_true[:, :0, :, :], y_true[:, 1:, :, :]), dim=1)

        # print(y_pred_fc.shape)

        # print(y_pred.shape)

        focal_loss = self.focal_loss(y_pred, torch.argmax(y_true, dim = 1)) #torchvision.ops.sigmoid_focal_loss(y_pred_fc, y_true_fc, reduction = "mean") #self.focal_loss(y_pred, y_true)
        iou_loss = 1 - self.iou_loss(y_pred, y_true)
        ms_ssim_loss = 1 - self.ms_ssim_loss(y_pred, y_true)

        # print(f"{focal_loss}, {iou_loss}, {ms_ssim_loss}")

        loss = (focal_loss + iou_loss + ms_ssim_loss) * weight #(dice_loss * self.weight) + (cross_entropy_loss * (1 - self.weight)) #focal_loss + iou_loss #+ ms_ssim_loss

        # loss = 0.0

        # for i, pred in enumerate(y_pred):

        #     pred = pred.to(self.device)
        #     # print(pred.shape)

        #     w = 1 / (2 ** i)

        #     focal_loss = self.focal_loss(pred, torch.argmax(y_true, dim = 1))
        #     iou_loss = 1 - self.iou_loss(pred, y_true)
        #     ms_ssim_loss = 1 - self.ms_ssim_loss(pred, y_true)

        #     loss += torch.sum((focal_loss + iou_loss + ms_ssim_loss) * w)

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

class FocalLoss_1(nn.Module):
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

class Focal_Test(nn.Module):
    def __init__(self):
        super().__init__()

        # self.focal_loss = FocalLossWithLogits(reduction="elementwise_mean")

        self.focal_loss = FocalLoss() #ignore_index = 0)

    def forward(self, y_pred, y_true):

        focal_loss = self.focal_loss(y_pred, y_true)

        return focal_loss

class IOULoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.epsilon = 1e-6

    def forward(self, y_pred, y_true):

        # y_pred = torch.nn.functional.softmax(y_pred)

        y_pred_flat = y_pred.flatten(start_dim = 1)
        y_true_flat = y_true.flatten(start_dim = 1)

        intersection = torch.sum(y_pred_flat * y_true_flat)
        union = y_pred_flat.sum() + y_true_flat.sum() - intersection

        iou = (intersection + self.epsilon) / (union + self.epsilon)

        thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10

        return 1 - iou

class IOU_Test(nn.Module):
    def __init__(self):
        super().__init__()

        self.iou = JaccardIndex(task = "multiclass", num_classes = 9) #, ignore_index = 0)
        # self.iou = MeanIoU(num_classes = 9, include_background = False)

    def _iou(self, y_pred, y_true):

        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        y_true = torch.argmax(y_true, dim = 1)

        iou = self.iou(y_pred, y_true)
        # iou = 1 - iou_loss

        return iou

    def forward(self, y_pred, y_true):
        
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        # y_pred = torch.argmax(y_pred, dim = 1)
        y_true = torch.argmax(y_true, dim = 1)

        return self.iou(y_pred, y_true)

class MS_SSIM_Test(nn.Module):
    def __init__(self):
        super().__init__()

        # self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(normalize='simple')
        self.ms_ssim = MS_SSIM(data_range = 1, channel = 9) #, weights = [0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])

    def _msssim(self, y_pred, y_true):

        y_pred = torch.nn.functional.softmax(y_pred, dim=1)

        ms_ssim = self.ms_ssim(y_pred, y_true)
        # ms_ssim = 1 - ms_ssim_loss

        return ms_ssim

    def forward(self, y_pred, y_true):
        
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        # print(y_pred[1:].shape)
        # print(y_pred[:4].shape)

        # y_pred = torch.cat((y_pred[:, :0, :, :], y_pred[:, 1:, :, :]), dim=1)
        # y_true = torch.cat((y_true[:, :0, :, :], y_true[:, 1:, :, :]), dim=1)

        return self.ms_ssim(y_pred, y_true)

class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=9):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        # self.weights = torch.autograd.Variable(weight * weight)
        self.epsilon = 1e-6

    def forward(self, y_pred, y_true, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        N = y_pred.size()[0]
        # if is_target_variable:
        #     target_oneHot = to_one_hot_var(target.data, self.classes).float()
        # else:
        #     target_oneHot = to_one_hot_var(target, self.classes).float()

        # predicted probabilities for each pixel along channel
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)

        # Numerator Product
        inter = y_pred * y_true
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = y_pred + y_true - (y_pred * y_true)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = (inter + self.epsilon) / (union + self.epsilon)

        ## Return average loss over classes and batch
        return -torch.mean(loss)

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def _iou(self, y_pred, y_true, size_average = True):

        y_pred = torch.nn.functional.softmax(y_pred, dim=1)

        b = y_pred.shape[0]
        IoU = 0.0
        for i in range(0,b):
            #compute the IoU of the foreground
            Iand1 = torch.sum(y_true[i,:,:,:]*y_pred[i,:,:,:])
            Ior1 = torch.sum(y_true[i,:,:,:]) + torch.sum(y_pred[i,:,:,:])-Iand1

            # y_pred_flat = y_pred.flatten(start_dim = 1)
            # y_true_flat = y_true.flatten(start_dim = 1)

            intersection = torch.sum(y_pred * y_true)
            union = y_pred.sum() + y_true.sum() - intersection

            IoU1 = intersection/union #Iand1/Ior1

            #IoU loss is (1-IoU1)
            IoU = IoU + (1-IoU1)

        return IoU/b

    def forward(self, pred, target):

        return self._iou(pred, target, self.size_average)

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

        L = 1

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
        cs = torch.mean(v1 / v2)

        ssim_map = ((2 * mu1_mu2 + self.C1) * v1) / ((mu1_sq + mu2_sq + self.C1) * v2)

        return ssim_map.mean(), cs

    def msssim(self, y_pred, y_true, window_size=11, size_average=True, val_range=None, normalize=None):
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(self.device)
        # weights = torch.FloatTensor([0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]).to(self.device)
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

        #Normalise
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

        pow1 = mcs **weights
        pow2 = ssims **weights

        return torch.prod(pow1[:-1] * pow2[-1])

    def forward(self, y_pred, y_true):

        y_pred = torch.nn.functional.softmax(y_pred)

        return 1 - self.msssim(y_pred, y_true, window_size = self.window_size, size_average = self.size_average)

class MS_SSIM_1(nn.Module):
    def __init__(self):
        super().__init__()

        self.ms_ssim = SSIM(in_channels = 9, return_msssim = True).cuda()

    def _msssim(self, y_pred, y_true):

        y_pred = torch.nn.functional.softmax(y_pred, dim=1)

        ms_ssim = self.ms_ssim(y_pred, y_true)

        return ms_ssim
    
    def forward(self, y_pred, y_true):

        return 1 - self._msssim(y_pred, y_true)

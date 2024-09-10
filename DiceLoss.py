import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, epsilon = 1e-6):

        predictions = nn.Softmax(dim = 1)(y_pred)

        # intersection = (y_true * predictions).sum()
        intersection = torch.sum(predictions * y_true) #, dim = 1)
        union = torch.sum(predictions + y_true) #, dim = 1)

        dice_coeff = (2. * intersection + epsilon) / (union + epsilon)
        dice_loss = 1 - dice_coeff

        dice_loss = torch.mean(dice_loss)

        # mod_a = intersection.sum()
        # mod_b = y_true.numel()

        # dice_coeff = 2. * intersection / (mod_a + mod_b + epsilon)
        # dice_loss = -dice_coeff.log()

        return dice_loss


import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicir.models.losses.loss_util import weighted_loss
from pytorch_msssim import ssim

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)

@weighted_loss
def ssim_loss(pred, target):
    return 1 - ssim(pred, target, data_range=1.0, size_average=False)

class SSIMLoss(nn.Module):
    """Structural Similarity Index (SSIM) loss.

    Args:
        loss_weight (float): Loss weight for SSIM loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SSIMLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * ssim_loss(
            pred, target, weight, reduction=self.reduction)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class SaturatedLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SaturatedLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, x):
        if self.reduction == 'mean':
            positive_saturation = torch.mean(torch.max(x, torch.tensor(1.0).to(x.device)))
            negative_saturation = torch.mean(torch.min(x, torch.tensor(0.0).to(x.device)))
            loss = positive_saturation - negative_saturation
        elif self.reduction == 'sum':
            positive_saturation = torch.sum(torch.max(x, torch.tensor(1.0).to(x.device)))
            negative_saturation = torch.sum(torch.min(x, torch.tensor(0.0).to(x.device)))
            loss = positive_saturation - negative_saturation
        return loss

class ColorCastLoss(nn.Module):
    def __init__(self):
        super(ColorCastLoss, self).__init__()
        # Define channel pairs (R, G), (G, B), (B, R)
        self.channel_pairs = [(0, 1), (1, 2), (2, 0)]

    def forward(self, enhanced_image):
        """
        Args:
            enhanced_image (torch.Tensor): A tensor of shape (N, C, H, W), where
                                           N = batch size,
                                           C = number of channels (3 for RGB),
                                           H, W = height and width of the image.

        Returns:
            torch.Tensor: The computed color cast loss (scalar).
        """
        # Ensure the input is a 4D tensor
        assert enhanced_image.ndim == 4, "Input must be a 4D tensor (N, C, H, W)"
        assert enhanced_image.size(1) == 3, "Input must have 3 color channels (RGB)"
        
        loss = 0.0
        # Iterate over channel pairs (R, G), (G, B), (B, R)
        for c1, c2 in self.channel_pairs:
            # Compute the mean of each channel
            mean_c1 = enhanced_image[:, c1, :, :].mean(dim=(1, 2))  # Mean of channel c1
            mean_c2 = enhanced_image[:, c2, :, :].mean(dim=(1, 2))  # Mean of channel c2
            
            # Compute the squared difference between the means
            loss += torch.mean((mean_c1 - mean_c2) ** 2)
        
        loss = loss / 3.0
        return loss

class ReconstructionLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ReconstructionLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, target, J, T, B):
        I_hat = J*T + B
        if self.reduction == 'mean':
            loss = F.l1_loss(I_hat, target, reduction='mean')
        elif self.reduction == 'sum':
            loss = F.l1_loss(I_hat, target, reduction='none')
        return loss

"""
Loss functions for virtual staining regression and GAN training.

Includes:
- L1Loss: Pixel-wise absolute error (robust, preserves sharpness)
- MSSSIMLoss: Multi-Scale Structural Similarity (perceptual quality)
- PerceptualLoss: VGG feature matching (texture fidelity)
- GANLoss: Adversarial loss for Pix2PixHD
- CombinedLoss: Weighted combination of any losses above
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from math import exp


# ---------------------------------------------------------------------------
# SSIM / MS-SSIM helpers
# ---------------------------------------------------------------------------

def _gaussian_window(size, sigma):
    """Create 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _create_window(window_size, channel):
    """Create 2D Gaussian window for SSIM."""
    _1d = _gaussian_window(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Compute SSIM between two images."""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)


class SSIMFunction(nn.Module):
    """Compute SSIM as a differentiable module."""

    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        channel = img1.size(1)
        window = _create_window(self.window_size, channel).to(img1.device).type_as(img1)
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class MSSSIMLoss(nn.Module):
    """Multi-Scale SSIM loss.

    Computes SSIM at multiple resolutions and combines them.
    Loss = 1 - MS_SSIM (so minimizing pushes SSIM toward 1).
    """

    def __init__(self, window_size=11, weights=None):
        super().__init__()
        self.window_size = window_size
        if weights is None:
            self.weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        else:
            self.weights = weights

    def forward(self, pred, target):
        channel = pred.size(1)
        window = _create_window(self.window_size, channel).to(pred.device).type_as(pred)

        levels = len(self.weights)
        mssim = []
        mcs = []

        for i in range(levels):
            ssim_val, cs_val = self._ssim_components(pred, target, window, channel)
            mssim.append(ssim_val)
            mcs.append(cs_val)

            if i < levels - 1:
                pred = F.avg_pool2d(pred, 2)
                target = F.avg_pool2d(target, 2)
                if pred.size(-1) < self.window_size:
                    break

        # Weight and combine (clamp to avoid NaN from negative ** fraction)
        n = len(mcs)
        weights = torch.tensor(self.weights[:n], device=pred.device)
        weights = weights / weights.sum()

        mcs_tensor = torch.stack(mcs).clamp(min=1e-8)
        mssim_tensor = torch.stack(mssim).clamp(min=1e-8)

        result = torch.prod(mcs_tensor[:-1] ** weights[:-1]) * (mssim_tensor[-1] ** weights[-1])

        return 1.0 - result.clamp(min=0.0, max=1.0)

    def _ssim_components(self, img1, img2, window, channel):
        """Return SSIM and contrast sensitivity separately."""
        ws = self.window_size
        mu1 = F.conv2d(img1, window, padding=ws // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=ws // 2, groups=channel)

        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=ws // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=ws // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=ws // 2, groups=channel) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

        return ssim_map.mean(), cs_map.mean()


# ---------------------------------------------------------------------------
# L1 Loss
# ---------------------------------------------------------------------------

class L1Loss(nn.Module):
    """Standard L1 (mean absolute error) loss."""

    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class WeightedL1Loss(nn.Module):
    """L1 loss that upweights non-zero (signal) pixels.

    For sparse IF targets (>98% black), standard L1 barely penalizes
    missing the rare bright pixels. This loss applies a higher weight
    to pixels where the target has signal.

    Args:
        signal_weight: Multiplier for pixels where target > threshold.
        bg_weight: Multiplier for background pixels.
        threshold: Value above which a pixel is considered signal.
    """

    def __init__(self, signal_weight=10.0, bg_weight=1.0, threshold=0.01):
        super().__init__()
        self.signal_weight = signal_weight
        self.bg_weight = bg_weight
        self.threshold = threshold

    def forward(self, pred, target):
        pixel_loss = torch.abs(pred - target)

        # Create weight map: high weight where target has signal
        weight = torch.where(
            target > self.threshold,
            torch.full_like(target, self.signal_weight),
            torch.full_like(target, self.bg_weight),
        )

        return (pixel_loss * weight).mean()


class FocalL1Loss(nn.Module):
    """Focal L1 loss — upweights pixels where the model is currently wrong.

    Unlike WeightedL1 which uses a fixed mask based on target values,
    FocalL1 dynamically focuses on hard-to-predict pixels. Pixels with
    large error get exponentially more weight during training.

    Loss = mean( |pred - target|^gamma * |pred - target| )
         = mean( |pred - target|^(gamma + 1) )

    gamma=0 -> standard L1
    gamma=1 -> squared L1 (focuses on large errors)
    gamma=2 -> cubed L1 (strongly focuses on large errors)

    Args:
        gamma: Focusing parameter. Higher = more focus on hard pixels.
        signal_boost: Extra multiplier for pixels where target > threshold.
    """

    def __init__(self, gamma=1.0, signal_boost=1.0, threshold=0.005):
        super().__init__()
        self.gamma = gamma
        self.signal_boost = signal_boost
        self.threshold = threshold

    def forward(self, pred, target):
        error = torch.abs(pred - target)
        focal_weight = error.detach() ** self.gamma  # Detach so gradient only flows through error

        loss = focal_weight * error

        if self.signal_boost > 1.0:
            signal_mask = (target > self.threshold).float()
            boost = 1.0 + (self.signal_boost - 1.0) * signal_mask
            loss = loss * boost

        return loss.mean()


class SoftDiceLoss(nn.Module):
    """Soft Dice loss for continuous (regression) predictions.

    Adapted from segmentation Dice loss to work with continuous targets.
    Measures overlap between predicted and target intensity distributions.
    Strongly penalizes missing signal regions — complementary to L1.

    Dice = 2 * sum(pred * target) / (sum(pred^2) + sum(target^2) + eps)
    Loss = 1 - Dice

    Args:
        smooth: Smoothing constant to avoid division by zero.
        per_channel: If True, compute Dice per channel and average.
    """

    def __init__(self, smooth=1.0, per_channel=True):
        super().__init__()
        self.smooth = smooth
        self.per_channel = per_channel

    def forward(self, pred, target):
        if self.per_channel:
            dice_sum = 0.0
            n_channels = pred.size(1)
            for c in range(n_channels):
                p = pred[:, c].reshape(pred.size(0), -1)
                t = target[:, c].reshape(target.size(0), -1)
                intersection = (p * t).sum(dim=1)
                union = (p ** 2).sum(dim=1) + (t ** 2).sum(dim=1)
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                dice_sum += dice.mean()
            return 1.0 - dice_sum / n_channels
        else:
            p = pred.reshape(pred.size(0), -1)
            t = target.reshape(target.size(0), -1)
            intersection = (p * t).sum(dim=1)
            union = (p ** 2).sum(dim=1) + (t ** 2).sum(dim=1)
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# Perceptual (VGG) Loss
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss.

    Compares activations at multiple VGG19 layers between pred and target.
    Good for capturing texture and structural similarity beyond pixel-level.

    For single-channel IF targets, we replicate to 3 channels before
    feeding through VGG.
    """

    def __init__(self, layers=None, weights=None):
        super().__init__()
        if layers is None:
            layers = [3, 8, 17, 26]  # relu1_2, relu2_2, relu3_4, relu4_4
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0]

        self.weights = weights
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slices = nn.ModuleList()

        prev = 0
        for layer_idx in layers:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:layer_idx + 1]))
            prev = layer_idx + 1

        # Freeze VGG weights
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _to_3ch(self, x):
        """Expand to 3 channels if needed."""
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) == 2:
            x = torch.cat([x, x[:, :1]], dim=1)
        return x

    def forward(self, pred, target):
        pred = self._to_3ch(pred)
        target = self._to_3ch(target)

        # Normalize to ImageNet stats
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0
        x_pred, x_target = pred, target
        for i, s in enumerate(self.slices):
            x_pred = s(x_pred)
            x_target = s(x_target)
            loss += self.weights[i] * F.l1_loss(x_pred, x_target)

        return loss


# ---------------------------------------------------------------------------
# GAN Losses
# ---------------------------------------------------------------------------

class GANLoss(nn.Module):
    """GAN loss supporting LSGAN (MSE) and vanilla (BCE) modes."""

    def __init__(self, mode="lsgan"):
        super().__init__()
        self.mode = mode
        if mode == "lsgan":
            self.loss_fn = nn.MSELoss()
        elif mode == "vanilla":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown GAN mode: {mode}")

    def forward(self, pred, is_real):
        if is_real:
            target = torch.ones_like(pred)
        else:
            target = torch.zeros_like(pred)
        return self.loss_fn(pred, target)


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for multi-scale discriminator.

    Matches intermediate features between real and fake across
    all discriminator layers. Stabilizes GAN training.
    """

    def __init__(self):
        super().__init__()

    def forward(self, real_features, fake_features):
        loss = 0.0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(fake_feat, real_feat.detach())
        return loss


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """Weighted combination of multiple losses.

    Args:
        loss_configs: list of dicts with keys 'name', 'weight', and optional params.
            e.g. [
                {"name": "l1", "weight": 1.0},
                {"name": "ms_ssim", "weight": 0.1},
                {"name": "perceptual", "weight": 0.05},
            ]
    """

    LOSS_MAP = {
        "l1": L1Loss,
        "weighted_l1": WeightedL1Loss,
        "focal_l1": FocalL1Loss,
        "dice": SoftDiceLoss,
        "ms_ssim": MSSSIMLoss,
        "perceptual": PerceptualLoss,
    }

    def __init__(self, loss_configs):
        super().__init__()
        self.losses = nn.ModuleList()
        self.weights = []
        self.names = []

        for cfg in loss_configs:
            name = cfg["name"]
            weight = cfg.get("weight", 1.0)
            params = cfg.get("params", {})

            if name not in self.LOSS_MAP:
                raise ValueError(f"Unknown loss: {name}. Available: {list(self.LOSS_MAP.keys())}")

            self.losses.append(self.LOSS_MAP[name](**params))
            self.weights.append(weight)
            self.names.append(name)

    def forward(self, pred, target):
        total = 0.0
        breakdown = {}
        for loss_fn, weight, name in zip(self.losses, self.weights, self.names):
            val = loss_fn(pred, target)
            total += weight * val
            breakdown[name] = val.item()
        return total, breakdown


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_loss(config):
    """Build loss function from config dict.

    Config expected keys:
        loss.type: 'l1', 'ms_ssim', 'perceptual', or 'combined'
        loss.losses: list of loss configs (for 'combined' type)
        loss.params: dict of params (for single loss types)
    """
    loss_cfg = config["loss"]
    loss_type = loss_cfg.get("type", "combined")

    if loss_type == "combined":
        return CombinedLoss(loss_cfg["losses"])
    elif loss_type in CombinedLoss.LOSS_MAP:
        params = loss_cfg.get("params", {})
        return CombinedLoss.LOSS_MAP[loss_type](**params)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

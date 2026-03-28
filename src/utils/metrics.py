"""
Evaluation metrics for virtual staining quality assessment.

Metrics:
- PSNR: Peak Signal-to-Noise Ratio (higher = better)
- SSIM: Structural Similarity Index (higher = better)
- LPIPS: Learned Perceptual Image Patch Similarity (lower = better)
- MAE: Mean Absolute Error (lower = better)
- PCC: Pearson Correlation Coefficient (higher = better)
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from ..losses.losses import SSIMFunction


def psnr(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(float("inf"))
    return 10 * torch.log10(max_val ** 2 / mse)


def ssim(pred, target, window_size=11):
    """Structural Similarity Index."""
    ssim_fn = SSIMFunction(window_size=window_size)
    return ssim_fn(pred, target)


def mae(pred, target):
    """Mean Absolute Error."""
    return F.l1_loss(pred, target)


def pearson_correlation(pred, target):
    """Pearson Correlation Coefficient (per-image, averaged over batch)."""
    batch_size = pred.size(0)
    pcc_vals = []
    for i in range(batch_size):
        p = pred[i].flatten()
        t = target[i].flatten()
        p_mean = p.mean()
        t_mean = t.mean()
        p_centered = p - p_mean
        t_centered = t - t_mean
        num = (p_centered * t_centered).sum()
        denom = torch.sqrt((p_centered ** 2).sum() * (t_centered ** 2).sum())
        if denom < 1e-8:
            pcc_vals.append(torch.tensor(0.0))
        else:
            pcc_vals.append(num / denom)
    return torch.stack(pcc_vals).mean()


def compute_metrics(pred, target):
    """Compute all metrics for a batch of predictions.

    Args:
        pred: [B, C, H, W] predicted image tensor in [0, 1]
        target: [B, C, H, W] ground truth image tensor in [0, 1]

    Returns:
        dict of metric_name -> value (float)
    """
    pred = pred.detach().clamp(0, 1)
    target = target.detach().clamp(0, 1)

    metrics = {
        "psnr": psnr(pred, target).item(),
        "ssim": ssim(pred, target).item(),
        "mae": mae(pred, target).item(),
        "pcc": pearson_correlation(pred, target).item(),
    }

    return metrics


class MetricTracker:
    """Accumulate metrics over batches and compute epoch-level averages."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = defaultdict(list)

    def update(self, metrics_dict):
        for k, v in metrics_dict.items():
            self.values[k].append(v)

    def summary(self):
        return {k: np.mean(v) for k, v in self.values.items()}

    def __str__(self):
        s = self.summary()
        parts = [f"{k}: {v:.4f}" for k, v in s.items()]
        return " | ".join(parts)

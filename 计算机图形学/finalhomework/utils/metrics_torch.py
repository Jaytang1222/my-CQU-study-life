"""
PyTorch-based metric utilities for fallback training.
"""
import torch
import numpy as np


def mse(pred, target):
    return float(torch.mean((pred - target) ** 2).item())


def mae(pred, target):
    return float(torch.mean(torch.abs(pred - target)).item())


def psnr(pred, target, max_val=1.0):
    mse_val = float(torch.mean((pred - target) ** 2).item())
    if mse_val == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse_val))


def ssim(pred, target):
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    mu1, mu2 = p.mean(), t.mean()
    sigma1 = ((p - mu1) ** 2).mean()
    sigma2 = ((t - mu2) ** 2).mean()
    sigma12 = ((p - mu1) * (t - mu2)).mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    )


def compute_metrics(pred, target, max_val=1.0):
    return {
        'mse': mse(pred, target),
        'mae': mae(pred, target),
        'psnr': psnr(pred, target, max_val=max_val),
        'ssim': ssim(pred, target),
    }


def print_metrics(metrics, prefix=''):
    if prefix:
        print(prefix, end=' ')
    print(
        f"MSE: {metrics['mse']:.6f}, "
        f"MAE: {metrics['mae']:.6f}, "
        f"PSNR: {metrics['psnr']:.2f}dB, "
        f"SSIM: {metrics['ssim']:.4f}"
    )

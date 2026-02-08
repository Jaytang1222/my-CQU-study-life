"""
Minimal PyTorch loss wrappers for fallback training.
Only a subset of losses (L1, L2/MSE) are provided for fallback.
"""
import torch
import torch.nn as nn


class L1Loss(nn.Module):
    def forward(self, pred, target):
        return nn.functional.l1_loss(pred, target)


class L2Loss(nn.Module):
    def forward(self, pred, target):
        return nn.functional.mse_loss(pred, target)


def get_loss_function(loss_type="l2", weights=None):
    loss_type = loss_type.lower()
    if loss_type == 'l1':
        return L1Loss()
    if loss_type == 'l2':
        return L2Loss()
    # Default fallback: mse
    return L2Loss()

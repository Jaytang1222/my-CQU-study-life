"""models package: provide backend-agnostic access to UNet

This module attempts to import the Jittor UNet first; if it fails, it falls
back to the PyTorch implementation.
"""
from importlib import import_module
_jt_unet = None
_torch_unet = None
try:
	mod = import_module('.unet', 'models')
	_jt_unet = mod.UNet
except Exception:
	_jt_unet = None

try:
	mod = import_module('.unet_torch', 'models')
	_torch_unet = mod.UNet
except Exception:
	_torch_unet = None


def get_unet(backend='jittor'):
	if backend == 'jittor' and _jt_unet is not None:
		return _jt_unet
	if backend == 'torch' and _torch_unet is not None:
		return _torch_unet
	# fallback logic: prefer jittor if available else torch
	if _jt_unet is not None:
		return _jt_unet
	if _torch_unet is not None:
		return _torch_unet
	raise RuntimeError('No UNet implementation available for jittor or torch')


__all__ = ['get_unet']



"""
数据增强与变换（numpy实现）
Transforms accept and return numpy arrays in CHW format (float32, range [0,1]).
"""
import numpy as np
from PIL import Image


class RandomCrop:
    def __init__(self, size):
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, noisy, clean):
        H, W = noisy.shape[1], noisy.shape[2]
        th, tw = self.h, self.w
        if H < th or W < tw:
            pad_h = max(0, th - H)
            pad_w = max(0, tw - W)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            # Pad with zeros
            noisy = np.pad(noisy, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
            clean = np.pad(clean, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
            H, W = noisy.shape[1], noisy.shape[2]
        i = np.random.randint(0, max(1, H - th + 1))
        j = np.random.randint(0, max(1, W - tw + 1))
        return noisy[:, i:i+th, j:j+tw], clean[:, i:i+th, j:j+tw]


class ResizeTo:
    def __init__(self, size):
        if isinstance(size, (int, float, np.integer)):
            self.h, self.w = int(size), int(size)
        else:
            self.h, self.w = size

    def __call__(self, noisy, clean):
        # noisy/clean are CHW numpy arrays in [0,1]
        def _resize(arr, h, w):
            c = arr.shape[0]
            img = (arr.transpose(1, 2, 0) * 255.0).astype(np.uint8)
            img = Image.fromarray(img)
            img = img.resize((w, h), Image.BILINEAR)
            out = np.array(img).astype(np.float32) / 255.0
            out = out.transpose(2, 0, 1)
            return out
        noisy_b = _resize(noisy, self.h, self.w)
        clean_b = _resize(clean, self.h, self.w)
        return noisy_b, clean_b


class MultiScaleResizeRandomCrop:
    def __init__(self, scales, crop_size):
        self.scales = scales
        self.cropper = RandomCrop(crop_size)

    def __call__(self, noisy, clean):
        size = int(np.random.choice(self.scales))
        noisy, clean = ResizeTo(size)(noisy, clean)
        noisy, clean = self.cropper(noisy, clean)
        return noisy, clean


class RandomFlip:
    def __init__(self, horizontal=True, vertical=False):
        self.h = horizontal
        self.v = vertical

    def __call__(self, noisy, clean):
        if self.h and np.random.rand() > 0.5:
            noisy = np.flip(noisy, axis=2).copy()
            clean = np.flip(clean, axis=2).copy()
        if self.v and np.random.rand() > 0.5:
            noisy = np.flip(noisy, axis=1).copy()
            clean = np.flip(clean, axis=1).copy()
        return noisy, clean


class RandomRotate:
    def __call__(self, noisy, clean):
        k = np.random.randint(0, 4)
        if k == 0:
            return noisy, clean
        def rot90(x, k):
            # x: C,H,W
            arr = np.rot90(x, k, axes=(1, 2)).copy()
            return arr
        return rot90(noisy, k), rot90(clean, k)


class Normalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean if mean is not None else [0.5, 0.5, 0.5]
        self.std = std if std is not None else [0.5, 0.5, 0.5]

    def __call__(self, noisy, clean):
        mean = np.array(self.mean).reshape(3, 1, 1).astype(np.float32)
        std = np.array(self.std).reshape(3, 1, 1).astype(np.float32)
        noisy = (noisy - mean) / std
        clean = (clean - mean) / std
        return noisy, clean


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, noisy, clean):
        for t in self.transforms:
            noisy, clean = t(noisy, clean)
        return noisy, clean


def get_train_transforms(crop_size=256, enable_multi_scale=False, multi_scales=None, use_crop=True):
    tfs = []
    if enable_multi_scale and multi_scales and use_crop:
        tfs.append(MultiScaleResizeRandomCrop(multi_scales, crop_size))
    else:
        tfs.append(ResizeTo(crop_size))
        if use_crop:
            tfs.append(RandomCrop(crop_size))
    tfs.extend([
        RandomFlip(horizontal=True, vertical=False),
        RandomRotate(),
        # Normalize()
    ])
    return Compose(tfs)


def get_val_transforms(crop_size=256):
    return Compose([ResizeTo(crop_size)])

"""
数据增强与变换
"""
import jittor as jt
import numpy as np


class RandomCrop:
    def __init__(self, size):
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, noisy, clean):
        H, W = noisy.shape[1], noisy.shape[2]
        th, tw = self.h, self.w
        if H < th or W < tw:
            pad_h = max(0, th - H)
            pad_w = max(0, tw - W)
            noisy = jt.nn.pad(noisy, [0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
            clean = jt.nn.pad(clean, [0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
            H, W = noisy.shape[1], noisy.shape[2]
        i = np.random.randint(0, H - th + 1)
        j = np.random.randint(0, W - tw + 1)
        return noisy[:, i:i+th, j:j+tw], clean[:, i:i+th, j:j+tw]


class ResizeTo:
    """将图像缩放到指定大小 (H, W)"""
    def __init__(self, size):
        # 兼容 numpy.int64 等标量
        if isinstance(size, (int, float, np.integer)):
            self.h, self.w = int(size), int(size)
        else:
            self.h, self.w = size

    def __call__(self, noisy, clean):
        # 输入为 CHW，插值要求 NCHW，这里临时加 batch 维度再去掉
        noisy_b = jt.nn.interpolate(noisy.unsqueeze(0), size=[self.h, self.w], mode="bilinear", align_corners=False)
        clean_b = jt.nn.interpolate(clean.unsqueeze(0), size=[self.h, self.w], mode="bilinear", align_corners=False)
        return noisy_b[0], clean_b[0]


class MultiScaleResizeRandomCrop:
    """多尺度随机缩放后再裁剪"""
    def __init__(self, scales, crop_size):
        self.scales = scales
        self.cropper = RandomCrop(crop_size)

    def __call__(self, noisy, clean):
        size = np.random.choice(self.scales)
        noisy, clean = ResizeTo(size)(noisy, clean)
        noisy, clean = self.cropper(noisy, clean)
        return noisy, clean


class RandomFlip:
    def __init__(self, horizontal=True, vertical=False):
        self.h = horizontal
        self.v = vertical

    def __call__(self, noisy, clean):
        if self.h and np.random.rand() > 0.5:
            noisy = jt.flip(noisy, dims=[2])
            clean = jt.flip(clean, dims=[2])
        if self.v and np.random.rand() > 0.5:
            noisy = jt.flip(noisy, dims=[1])
            clean = jt.flip(clean, dims=[1])
        return noisy, clean


class RandomRotate:
    def __call__(self, noisy, clean):
        k = np.random.randint(0, 4)
        if k == 0:
            return noisy, clean
        # 90 度旋转实现（CHW）
        def rot90(x, k):
            k = k % 4
            if k == 1:
                x = jt.transpose(x, (0, 2, 1))
                x = jt.flip(x, dims=[2])
            elif k == 2:
                x = jt.flip(x, dims=[1, 2])
            elif k == 3:
                x = jt.transpose(x, (0, 2, 1))
                x = jt.flip(x, dims=[1])
            return x
        return rot90(noisy, k), rot90(clean, k)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, noisy, clean):
        for t in self.transforms:
            noisy, clean = t(noisy, clean)
        return noisy, clean


def get_train_transforms(crop_size=256):
    return Compose([
        RandomCrop(crop_size),
        RandomFlip(horizontal=True, vertical=False),
        RandomRotate(),
    ])


def get_val_transforms():
    return Compose([])
"""
数据增强和变换
"""
import jittor as jt
import numpy as np


class RandomCrop:
    """随机裁剪"""
    
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)
    
    def __call__(self, noisy, clean):
        h, w = noisy.shape[1], noisy.shape[2]
        th, tw = self.size
        
        if h < th or w < tw:
            # 如果图像太小，进行填充
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)
            # jt.nn.pad格式: [pad_left, pad_right, pad_top, pad_bottom] for 2D
            # 对于3D [C,H,W]，需要: [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back]
            # 但我们只需要pad H和W，所以格式是: [0, 0, pad_top, pad_bottom, pad_left, pad_right]
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            noisy = jt.nn.pad(noisy, [0, 0, pad_left, pad_right, pad_top, pad_bottom])
            clean = jt.nn.pad(clean, [0, 0, pad_left, pad_right, pad_top, pad_bottom])
            # 更新尺寸
            h, w = noisy.shape[1], noisy.shape[2]
        
        # 确保裁剪范围有效
        if h < th or w < tw:
            # 如果填充后仍然不够，直接resize
            import jittor.nn as nn
            noisy = nn.interpolate(noisy.unsqueeze(0), size=(th, tw), mode='bilinear', align_corners=False)[0]
            clean = nn.interpolate(clean.unsqueeze(0), size=(th, tw), mode='bilinear', align_corners=False)[0]
            return noisy, clean
        
        i = np.random.randint(0, max(1, h - th + 1))
        j = np.random.randint(0, max(1, w - tw + 1))
        
        noisy = noisy[:, i:i+th, j:j+tw]
        clean = clean[:, i:i+th, j:j+tw]
        
        return noisy, clean


class RandomFlip:
    """随机翻转"""
    
    def __init__(self, horizontal=True, vertical=False):
        self.horizontal = horizontal
        self.vertical = vertical
    
    def __call__(self, noisy, clean):
        if self.horizontal and np.random.rand() > 0.5:
            noisy = jt.flip(noisy, dim=2)
            clean = jt.flip(clean, dim=2)
        
        if self.vertical and np.random.rand() > 0.5:
            noisy = jt.flip(noisy, dim=1)
            clean = jt.flip(clean, dim=1)
        
        return noisy, clean


class RandomRotate:
    """随机旋转（90度的倍数）"""
    
    def __init__(self):
        pass
    
    def __call__(self, noisy, clean):
        k = np.random.randint(0, 4)
        if k == 0:
            return noisy, clean
        # 自实现 90 度旋转 (CHW)
        def rot90(x, k):
            k = k % 4
            if k == 1:
                x = jt.transpose(x, (0, 2, 1))
                x = jt.flip(x, dim=2)
            elif k == 2:
                x = jt.flip(x, dim=1)
                x = jt.flip(x, dim=2)
            elif k == 3:
                x = jt.transpose(x, (0, 2, 1))
                x = jt.flip(x, dim=1)
            return x
        return rot90(noisy, k), rot90(clean, k)


class Normalize:
    """归一化"""
    
    def __init__(self, mean=None, std=None):
        self.mean = mean if mean is not None else [0.5, 0.5, 0.5]
        self.std = std if std is not None else [0.5, 0.5, 0.5]
    
    def __call__(self, noisy, clean):
        # 假设输入已经是 [0, 1] 范围
        # 转换为 [-1, 1]
        noisy = (noisy - jt.array(self.mean).view(3, 1, 1)) / jt.array(self.std).view(3, 1, 1)
        clean = (clean - jt.array(self.mean).view(3, 1, 1)) / jt.array(self.std).view(3, 1, 1)
        return noisy, clean


class Compose:
    """组合多个变换"""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, noisy, clean):
        for t in self.transforms:
            noisy, clean = t(noisy, clean)
        return noisy, clean


def get_train_transforms(crop_size=256, enable_multi_scale=False, multi_scales=None, use_crop=True):
    """获取训练时的数据增强
    Args:
        crop_size: 目标尺寸
        enable_multi_scale: 是否启用多尺度
        multi_scales: 多尺度列表
        use_crop: 是否使用RandomCrop（False时只resize到统一尺寸）
    """
    tfs = []
    if enable_multi_scale and multi_scales and use_crop:
        # 只有在use_crop=True时才使用MultiScaleResizeRandomCrop
        tfs.append(MultiScaleResizeRandomCrop(multi_scales, crop_size))
    else:
        tfs.append(ResizeTo(crop_size))
        if use_crop:
            tfs.append(RandomCrop(crop_size))
    tfs.extend([
        RandomFlip(horizontal=True, vertical=False),
        RandomRotate(),
        # Normalize()  # 根据需要启用
    ])
    return Compose(tfs)


def get_val_transforms(crop_size=256):
    """获取验证时的变换（通常不做增强）"""
    return Compose([
        ResizeTo(crop_size),
        # Normalize()  # 根据需要启用
    ])



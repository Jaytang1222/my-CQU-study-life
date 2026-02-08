import glob
import os
from typing import List, Tuple

import jittor as jt
from jittor.dataset import Dataset
import numpy as np
from PIL import Image


def _list_images(root: str) -> List[str]:
    exts = ["png", "jpg", "jpeg", "JPG", "PNG", "JPEG"]
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, f"**/*.{ext}"), recursive=True))
    return sorted(files)


def _normalize_stem(stem: str) -> str:
    """
    将文件名去掉后缀标记（如 _mean / _real / -mean / -real），用于配对。
    例如：
    - Canon5D2_..._chair_5_mean  -> Canon5D2_..._chair_5
    - Canon5D2_..._chair_5_real  -> Canon5D2_..._chair_5
    """
    suffixes = ["_mean", "_real", "-mean", "-real"]
    for suf in suffixes:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _match_pairs(clean_root: str, noisy_root: str) -> Tuple[List[str], List[str]]:
    clean_files = _list_images(clean_root)
    noisy_files = _list_images(noisy_root)
    if not clean_files:
        raise FileNotFoundError(f"未在 {clean_root} 找到 clean 图片")
    if not noisy_files:
        raise FileNotFoundError(f"未在 {noisy_root} 找到 noisy 图片")

    noisy_map = {
        _normalize_stem(os.path.splitext(os.path.basename(p))[0]): p for p in noisy_files
    }
    clean_paths, noisy_paths = [], []
    for c in clean_files:
        stem = _normalize_stem(os.path.splitext(os.path.basename(c))[0])
        if stem in noisy_map:
            clean_paths.append(c)
            noisy_paths.append(noisy_map[stem])
    if not clean_paths:
        raise RuntimeError("未能在 clean/noisy 目录中找到同名文件对")
    return clean_paths, noisy_paths


class MCDenoiseDataset(Dataset):
    """
    成对降噪数据集：
    - clean / noisy 目录中按同名文件配对
    - 输入: noisy + 样本数通道
    - 标签: clean
    """

    def __init__(
        self,
        clean_root: str,
        noisy_root: str,
        image_size: int,
        num_mc_samples: int,
        max_mc_samples: int,
        noise_sigma: float,
    ):
        super().__init__()
        self.clean_paths, self.noisy_paths = _match_pairs(clean_root, noisy_root)
        self.image_size = image_size
        self.num_mc_samples = num_mc_samples
        self.max_mc_samples = max_mc_samples
        self.noise_sigma = noise_sigma
        self.set_attrs(total_len=len(self.clean_paths))

    def __len__(self):
        return len(self.clean_paths)

    def _load_image(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        if self.image_size:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)  # C, H, W

    def __getitem__(self, index: int) -> Tuple[jt.Var, jt.Var]:
        clean_np = self._load_image(self.clean_paths[index])
        noisy_np = self._load_image(self.noisy_paths[index])

        clean = jt.array(clean_np)
        noisy = jt.array(noisy_np)

        # 可选：在真实 noisy 上再叠加微小噪声做增强
        if self.noise_sigma > 0:
            noise = jt.randn_like(noisy) * self.noise_sigma
            noisy = jt.clamp(noisy + noise, 0.0, 1.0)

        sample_ratio = self.num_mc_samples / float(self.max_mc_samples)
        sample_channel = jt.full((1, noisy.shape[1], noisy.shape[2]), sample_ratio)

        # 4 通道输入: noisy_rgb(3) + sample_count(1)
        model_input = jt.concat([noisy, sample_channel], dim=0)
        return model_input, clean


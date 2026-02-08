"""
Patch-based数据加载器：将大图分成多个patch进行训练
适用于大尺寸图像，可以保留更多细节
"""
import jittor as jt
import numpy as np
from dataset import MonteCarloDenoiseDataset


class PatchDataset:
    """将大图数据集转换为patch数据集"""
    
    def __init__(self, base_dataset, patch_size=512, patches_per_image=4, overlap=64):
        """
        Args:
            base_dataset: 基础数据集（MonteCarloDenoiseDataset）
            patch_size: patch大小
            patches_per_image: 每张图提取的patch数量
            overlap: patch之间的重叠像素
        """
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.overlap = overlap
        
        # 预计算所有patch的索引
        self.patches = []
        for idx in range(len(base_dataset)):
            # 获取图像尺寸（需要先加载一次）
            sample = base_dataset[idx]
            noisy = sample['noisy']
            h, w = noisy.shape[1], noisy.shape[2]
            
            # 如果图像小于patch_size，直接使用整张图
            if h < patch_size or w < patch_size:
                self.patches.append((idx, 0, 0))
            else:
                # 计算可能的patch位置
                step = patch_size - overlap
                h_steps = max(1, (h - overlap) // step)
                w_steps = max(1, (w - overlap) // step)
                
                # 随机选择patches_per_image个位置
                positions = []
                for hi in range(h_steps):
                    for wi in range(w_steps):
                        y = min(hi * step, h - patch_size)
                        x = min(wi * step, w - patch_size)
                        positions.append((y, x))
                
                # 随机选择或全部使用
                if len(positions) > patches_per_image:
                    import random
                    selected = random.sample(positions, patches_per_image)
                else:
                    selected = positions
                
                for y, x in selected:
                    self.patches.append((idx, y, x))
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, patch_idx):
        """获取一个patch"""
        img_idx, y, x = self.patches[patch_idx]
        sample = self.base_dataset[img_idx]
        
        noisy = sample['noisy']
        clean = sample['clean']
        filename = sample.get('filename', f'img_{img_idx}')
        
        h, w = noisy.shape[1], noisy.shape[2]
        
        # 提取patch
        if h >= self.patch_size and w >= self.patch_size:
            # 确保不越界
            y = min(int(y), int(h - self.patch_size))
            x = min(int(x), int(w - self.patch_size))
            noisy_patch = noisy[:, y:y+self.patch_size, x:x+self.patch_size]
            clean_patch = clean[:, y:y+self.patch_size, x:x+self.patch_size]
        else:
            # 如果图像太小，进行padding
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            noisy_patch = jt.nn.pad(
                noisy,
                [0, 0, pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2]
            )
            clean_patch = jt.nn.pad(
                clean,
                [0, 0, pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2]
            )
            # 如果padding后还是不够，裁剪
            noisy_patch = noisy_patch[:, :self.patch_size, :self.patch_size]
            clean_patch = clean_patch[:, :self.patch_size, :self.patch_size]
        
        return {
            'noisy': noisy_patch,
            'clean': clean_patch,
            'filename': f"{filename}_patch_{y}_{x}"
        }
    
    def set_attrs(self, **kwargs):
        """兼容Jittor Dataset接口"""
        self.base_dataset.set_attrs(**kwargs)


def create_patch_dataset(base_dataset, patch_size=512, patches_per_image=4, overlap=64):
    """创建patch数据集"""
    return PatchDataset(base_dataset, patch_size, patches_per_image, overlap)


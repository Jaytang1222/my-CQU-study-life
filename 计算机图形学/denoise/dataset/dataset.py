"""
Minimal dataset module (clean)
Returns numpy arrays (CHW, float32) for images and avoids importing jittor at module import time.
"""
import os
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MonteCarloDenoiseDataset(object):
    def __init__(self, data_dir, split='train', transform=None, load_auxiliary=False):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.load_auxiliary = load_auxiliary

        self.noisy_dir = os.path.join(data_dir, split, 'noisy_images')
        self.clean_dir = os.path.join(data_dir, split, 'clean_images')
        # support alternative names
        if not os.path.exists(self.noisy_dir):
            self.noisy_dir = os.path.join(data_dir, split, 'noisy')
        if not os.path.exists(self.clean_dir):
            self.clean_dir = os.path.join(data_dir, split, 'clean')

        self.aux_dir = os.path.join(data_dir, split, 'auxiliary') if load_auxiliary else None

        def list_images(path):
            if not os.path.exists(path):
                return []
            return sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.exr'))])

        self.noisy_files = list_images(self.noisy_dir)
        self.clean_files = list_images(self.clean_dir)
        if len(self.noisy_files) != len(self.clean_files):
            common = sorted(list(set(self.noisy_files) & set(self.clean_files)))
            self.noisy_files = common
            self.clean_files = common
        self.total_len = len(self.noisy_files)
        print(f"Loaded {self.total_len} image pairs for '{split}' split")

    def __len__(self):
        return self.total_len

    def _load_image(self, path):
        if path.lower().endswith('.exr'):
            try:
                import imageio
                img = imageio.imread(path).astype(np.float32)
            except Exception:
                img = np.zeros((256, 256, 3), dtype=np.float32)
        else:
            try:
                img = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
            except Exception:
                img = np.zeros((256, 256, 3), dtype=np.float32)
        if img.ndim == 3:
            img = img.transpose(2, 0, 1)
        else:
            img = img[np.newaxis, :, :]
        return img

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])
        noisy = self._load_image(noisy_path)
        clean = self._load_image(clean_path)
        if self.transform:
            noisy, clean = self.transform(noisy, clean)
        item = {'noisy': noisy, 'clean': clean, 'filename': self.noisy_files[idx]}
        if self.load_auxiliary and self.aux_dir:
            aux_path = os.path.join(self.aux_dir, self.noisy_files[idx])
            if os.path.exists(aux_path):
                item['auxiliary'] = self._load_image(aux_path)
        return item


def create_data_loader(dataset, batch_size=4, shuffle=True, num_workers=0, backend='py'):
    """Create simple loaders for py and torch. For jittor, return dataset and let train script adapt."""
    if backend == 'torch':
        try:
            import torch
            from torch.utils.data import DataLoader
        except Exception:
            raise RuntimeError("PyTorch is not installed but backend='torch' was requested")
        class _TorchDataset(torch.utils.data.Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                sample = self.ds[idx]
                t_noisy = torch.from_numpy(sample['noisy'].astype(np.float32))
                t_clean = torch.from_numpy(sample['clean'].astype(np.float32))
                item = {'noisy': t_noisy, 'clean': t_clean, 'filename': sample['filename']}
                if 'auxiliary' in sample:
                    item['auxiliary'] = torch.from_numpy(sample['auxiliary'].astype(np.float32))
                return item
        return DataLoader(_TorchDataset(dataset), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    elif backend == 'jittor':
        return dataset
    else:
        def _generator():
            idxs = list(range(len(dataset)))
            if shuffle:
                import random
                random.shuffle(idxs)
            batch = []
            for i in idxs:
                batch.append(dataset[i])
                if len(batch) >= batch_size:
                    noisy = np.stack([b['noisy'] for b in batch], axis=0)
                    clean = np.stack([b['clean'] for b in batch], axis=0)
                    """
                    Minimal dataset module (clean)
                    Returns numpy arrays (CHW, float32) for images and avoids importing jittor at module import time.
                    """
                    import os
                    import numpy as np
                    from PIL import Image, ImageFile
                    ImageFile.LOAD_TRUNCATED_IMAGES = True


                    class MonteCarloDenoiseDataset(object):
                        def __init__(self, data_dir, split='train', transform=None, load_auxiliary=False):
                            self.data_dir = data_dir
                            self.split = split
                            self.transform = transform
                            self.load_auxiliary = load_auxiliary

                            self.noisy_dir = os.path.join(data_dir, split, 'noisy_images')
                            self.clean_dir = os.path.join(data_dir, split, 'clean_images')
                            # support alternative names
                            if not os.path.exists(self.noisy_dir):
                                self.noisy_dir = os.path.join(data_dir, split, 'noisy')
                            if not os.path.exists(self.clean_dir):
                                self.clean_dir = os.path.join(data_dir, split, 'clean')

                            self.aux_dir = os.path.join(data_dir, split, 'auxiliary') if load_auxiliary else None

                            def list_images(path):
                                if not os.path.exists(path):
                                    return []
                                return sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.exr'))])

                            self.noisy_files = list_images(self.noisy_dir)
                            self.clean_files = list_images(self.clean_dir)
                            if len(self.noisy_files) != len(self.clean_files):
                                common = sorted(list(set(self.noisy_files) & set(self.clean_files)))
                                self.noisy_files = common
                                self.clean_files = common
                            self.total_len = len(self.noisy_files)
                            print(f"Loaded {self.total_len} image pairs for '{split}' split")

                        def __len__(self):
                            return self.total_len

                        def _load_image(self, path):
                            if path.lower().endswith('.exr'):
                                try:
                                    import imageio
                                    img = imageio.imread(path).astype(np.float32)
                                except Exception:
                                    img = np.zeros((256, 256, 3), dtype=np.float32)
                            else:
                                try:
                                    img = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
                                except Exception:
                                    img = np.zeros((256, 256, 3), dtype=np.float32)
                            if img.ndim == 3:
                                img = img.transpose(2, 0, 1)
                            else:
                                img = img[np.newaxis, :, :]
                            return img

                        def __getitem__(self, idx):
                            noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
                            clean_path = os.path.join(self.clean_dir, self.clean_files[idx])
                            noisy = self._load_image(noisy_path)
                            clean = self._load_image(clean_path)
                            if self.transform:
                                noisy, clean = self.transform(noisy, clean)
                            item = {'noisy': noisy, 'clean': clean, 'filename': self.noisy_files[idx]}
                            if self.load_auxiliary and self.aux_dir:
                                aux_path = os.path.join(self.aux_dir, self.noisy_files[idx])
                                if os.path.exists(aux_path):
                                    item['auxiliary'] = self._load_image(aux_path)
                            return item


                    def create_data_loader(dataset, batch_size=4, shuffle=True, num_workers=0, backend='py'):
                        """Create simple loaders for py and torch. For jittor, return dataset and let train script adapt."""
                        if backend == 'torch':
                            try:
                                import torch
                                from torch.utils.data import DataLoader
                            except Exception:
                                raise RuntimeError("PyTorch is not installed but backend='torch' was requested")
                            class _TorchDataset(torch.utils.data.Dataset):
                                def __init__(self, ds):
                                    self.ds = ds
                                def __len__(self):
                                    return len(self.ds)
                                def __getitem__(self, idx):
                                    sample = self.ds[idx]
                                    t_noisy = torch.from_numpy(sample['noisy'].astype(np.float32))
                                    t_clean = torch.from_numpy(sample['clean'].astype(np.float32))
                                    item = {'noisy': t_noisy, 'clean': t_clean, 'filename': sample['filename']}
                                    if 'auxiliary' in sample:
                                        item['auxiliary'] = torch.from_numpy(sample['auxiliary'].astype(np.float32))
                                    return item
                            return DataLoader(_TorchDataset(dataset), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
                        elif backend == 'jittor':
                            return dataset
                        else:
                            def _generator():
                                idxs = list(range(len(dataset)))
                                if shuffle:
                                    import random
                                    random.shuffle(idxs)
                                batch = []
                                for i in idxs:
                                    batch.append(dataset[i])
                                    if len(batch) >= batch_size:
                                        noisy = np.stack([b['noisy'] for b in batch], axis=0)
                                        clean = np.stack([b['clean'] for b in batch], axis=0)
                                        filenames = [b['filename'] for b in batch]
                                        yield {'noisy': noisy, 'clean': clean, 'filename': filenames}
                                        batch = []
                                if batch:
                                    noisy = np.stack([b['noisy'] for b in batch], axis=0)
                                    clean = np.stack([b['clean'] for b in batch], axis=0)
                                    filenames = [b['filename'] for b in batch]
                                    yield {'noisy': noisy, 'clean': clean, 'filename': filenames}
                            return _generator()
                    clean = np.stack([b['clean'] for b in batch], axis=0)
                    filenames = [b['filename'] for b in batch]
                    yield {'noisy': noisy, 'clean': clean, 'filename': filenames}
                    batch = []
            if batch:
                noisy = np.stack([b['noisy'] for b in batch], axis=0)
                clean = np.stack([b['clean'] for b in batch], axis=0)
                filenames = [b['filename'] for b in batch]
                yield {'noisy': noisy, 'clean': clean, 'filename': filenames}
        return _generator()
"""
MonteCarloDenoiseDataset - backend-agnostic dataset

This dataset avoids importing Jittor at module import time. It returns numpy
arrays (CHW, float32) from __getitem__. The training script / loader should
convert numpy arrays to framework-specific tensors (torch or jittor) when
needed.
"""
import os
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MonteCarloDenoiseDataset(object):
    """
    Monte Carlo denoise dataset. Returns dict with keys:
    - 'noisy': numpy array C,H,W float32
    - 'clean': numpy array C,H,W float32
    - 'filename': str
    - optional 'auxiliary'
    """
    def __init__(self, data_dir, split='train', transform=None, load_auxiliary=False):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.load_auxiliary = load_auxiliary

        candidate_noisy_dirs = [
            os.path.join(data_dir, split, 'noisy_images'),
            os.path.join(data_dir, split, 'noisy'),
        ]
        candidate_clean_dirs = [
            os.path.join(data_dir, split, 'clean_images'),
            os.path.join(data_dir, split, 'clean'),
        ]

        self.noisy_dir = next((d for d in candidate_noisy_dirs if os.path.exists(d)), candidate_noisy_dirs[0])
        self.clean_dir = next((d for d in candidate_clean_dirs if os.path.exists(d)), candidate_clean_dirs[0])
        self.aux_dir = os.path.join(data_dir, split, 'auxiliary') if load_auxiliary else None

        def list_images(path):
            if not os.path.exists(path):
                return []
            return sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.exr'))])

        self.noisy_files = list_images(self.noisy_dir)
        self.clean_files = list_images(self.clean_dir)

        if len(self.noisy_files) != len(self.clean_files):
            common = sorted(list(set(self.noisy_files) & set(self.clean_files)))
            self.noisy_files = common
            self.clean_files = common

        self.total_len = len(self.noisy_files)
        print(f"Loaded {self.total_len} image pairs for '{split}' split")

    def __len__(self):
        return self.total_len

    def load_image(self, path):
        if path.lower().endswith('.exr'):
            try:
                import imageio
                img = imageio.imread(path).astype(np.float32)
            except Exception:
                print(f"Warning: failed to read EXR {path}, using zeros")
                img = np.zeros((256, 256, 3), dtype=np.float32)
        else:
            try:
                img = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
            except Exception as e:
                print(f"Warning: cannot read image {path}: {e}; using zeros")
                img = np.zeros((256, 256, 3), dtype=np.float32)

        if img.ndim == 3:
            img = img.transpose(2, 0, 1)
        else:
            img = img[np.newaxis, :, :]
        return img

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])

        noisy = self.load_image(noisy_path)
        clean = self.load_image(clean_path)

        if self.transform:
            noisy, clean = self.transform(noisy, clean)

        item = {'noisy': noisy, 'clean': clean, 'filename': self.noisy_files[idx]}
        if self.load_auxiliary and self.aux_dir:
            aux_path = os.path.join(self.aux_dir, self.noisy_files[idx])
            if os.path.exists(aux_path):
                item['auxiliary'] = self.load_image(aux_path)
        return item


def create_data_loader(dataset, batch_size=4, shuffle=True, num_workers=0, backend='py'):
    if backend == 'torch':
        try:
            import torch
            from torch.utils.data import DataLoader
        except Exception:
            raise RuntimeError("PyTorch is not installed but backend='torch' was requested")

        class _TorchDataset(torch.utils.data.Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                sample = self.ds[idx]
                t_noisy = torch.from_numpy(sample['noisy'].astype(np.float32))
                t_clean = torch.from_numpy(sample['clean'].astype(np.float32))
                item = {'noisy': t_noisy, 'clean': t_clean, 'filename': sample['filename']}
                if 'auxiliary' in sample:
                    item['auxiliary'] = torch.from_numpy(sample['auxiliary'].astype(np.float32))
                return item

        return DataLoader(_TorchDataset(dataset), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    elif backend == 'jittor':
        return dataset
    else:
        def _generator():
            idxs = list(range(len(dataset)))
            if shuffle:
                import random
                random.shuffle(idxs)
            batch = []
            for i in idxs:
                batch.append(dataset[i])
                if len(batch) >= batch_size:
                    noisy = np.stack([b['noisy'] for b in batch], axis=0)
                    clean = np.stack([b['clean'] for b in batch], axis=0)
                    filenames = [b['filename'] for b in batch]
                    yield {'noisy': noisy, 'clean': clean, 'filename': filenames}
                    batch = []
            if batch:
                noisy = np.stack([b['noisy'] for b in batch], axis=0)
                clean = np.stack([b['clean'] for b in batch], axis=0)
                filenames = [b['filename'] for b in batch]
                yield {'noisy': noisy, 'clean': clean, 'filename': filenames}
        return _generator()
"""
MonteCarloDenoiseDataset - backend-agnostic dataset

This dataset avoids importing Jittor at module import time. It returns numpy
arrays (CHW, float32) from __getitem__. The training script / loader should
convert numpy arrays to framework-specific tensors (torch or jittor) when
needed.
"""
import os
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MonteCarloDenoiseDataset(object):
    """
    Monte Carlo denoise dataset. Returns dict with keys:
    - 'noisy': numpy array C,H,W float32
    - 'clean': numpy array C,H,W float32
    - 'filename': str
    - optional 'auxiliary'
    """
    def __init__(self, data_dir, split='train', transform=None, load_auxiliary=False):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.load_auxiliary = load_auxiliary

        candidate_noisy_dirs = [
            os.path.join(data_dir, split, 'noisy_images'),
            os.path.join(data_dir, split, 'noisy'),
        ]
        candidate_clean_dirs = [
            os.path.join(data_dir, split, 'clean_images'),
            os.path.join(data_dir, split, 'clean'),
        ]

        self.noisy_dir = next((d for d in candidate_noisy_dirs if os.path.exists(d)), candidate_noisy_dirs[0])
        self.clean_dir = next((d for d in candidate_clean_dirs if os.path.exists(d)), candidate_clean_dirs[0])
        self.aux_dir = os.path.join(data_dir, split, 'auxiliary') if load_auxiliary else None

        def list_images(path):
            if not os.path.exists(path):
                return []
            return sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.exr'))])

        self.noisy_files = list_images(self.noisy_dir)
        self.clean_files = list_images(self.clean_dir)

        if len(self.noisy_files) != len(self.clean_files):
            common = sorted(list(set(self.noisy_files) & set(self.clean_files)))
            self.noisy_files = common
            self.clean_files = common

        self.total_len = len(self.noisy_files)
        print(f"Loaded {self.total_len} image pairs for '{split}' split")

    def __len__(self):
        return self.total_len

    def load_image(self, path):
        if path.lower().endswith('.exr'):
            try:
                import imageio
                img = imageio.imread(path).astype(np.float32)
            except Exception:
                print(f"Warning: failed to read EXR {path}, using zeros")
                img = np.zeros((256, 256, 3), dtype=np.float32)
        else:
            try:
                img = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
            except Exception as e:
                print(f"Warning: cannot read image {path}: {e}; using zeros")
                img = np.zeros((256, 256, 3), dtype=np.float32)

        if img.ndim == 3:
            img = img.transpose(2, 0, 1)
        else:
            img = img[np.newaxis, :, :]
        return img

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])

        noisy = self.load_image(noisy_path)
        clean = self.load_image(clean_path)

        if self.transform:
            noisy, clean = self.transform(noisy, clean)

        item = {'noisy': noisy, 'clean': clean, 'filename': self.noisy_files[idx]}
        if self.load_auxiliary and self.aux_dir:
            aux_path = os.path.join(self.aux_dir, self.noisy_files[idx])
            if os.path.exists(aux_path):
                item['auxiliary'] = self.load_image(aux_path)
        return item


def create_data_loader(dataset, batch_size=4, shuffle=True, num_workers=0, backend='py'):
    if backend == 'torch':
        try:
            import torch
            from torch.utils.data import DataLoader
        except Exception:
            raise RuntimeError("PyTorch is not installed but backend='torch' was requested")

        class _TorchDataset(torch.utils.data.Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                sample = self.ds[idx]
                t_noisy = torch.from_numpy(sample['noisy'].astype(np.float32))
                t_clean = torch.from_numpy(sample['clean'].astype(np.float32))
                item = {'noisy': t_noisy, 'clean': t_clean, 'filename': sample['filename']}
                if 'auxiliary' in sample:
                    item['auxiliary'] = torch.from_numpy(sample['auxiliary'].astype(np.float32))
                return item

        return DataLoader(_TorchDataset(dataset), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    elif backend == 'jittor':
        # Return dataset object; Jittor-based loader should set dataset attributes in the training code
        return dataset
    else:
        def _generator():
            idxs = list(range(len(dataset)))
            if shuffle:
                import random
                random.shuffle(idxs)
            batch = []
            for i in idxs:
                batch.append(dataset[i])
                if len(batch) >= batch_size:
                    noisy = np.stack([b['noisy'] for b in batch], axis=0)
                    clean = np.stack([b['clean'] for b in batch], axis=0)
                    filenames = [b['filename'] for b in batch]
                    yield {'noisy': noisy, 'clean': clean, 'filename': filenames}
                    batch = []
            if batch:
                noisy = np.stack([b['noisy'] for b in batch], axis=0)
                clean = np.stack([b['clean'] for b in batch], axis=0)
                filenames = [b['filename'] for b in batch]
                yield {'noisy': noisy, 'clean': clean, 'filename': filenames}
        return _generator()
"""
MonteCarloDenoiseDataset - dataset module

This dataset intentionally avoids importing Jittor or Torch at module import time.
It returns numpy arrays (CHW, float32) from __getitem__, and leaves tensor
conversion to the training script, which decides whether to convert to
Jittor/torch tensors.

Supported folder naming conventions:
 - noisy/ / clean/
 - noisy_images/ / clean_images/
"""
import os
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MonteCarloDenoiseDataset(object):
    """
    蒙特卡洛渲染去噪数据集

    支持的数据目录结构（两种形式）：
    1) data_dir/train/noisy/ and data_dir/train/clean/
    2) data_dir/train/noisy_images/ and data_dir/train/clean_images/
    """

    def __init__(self, data_dir, split="train", transform=None, load_auxiliary=False):
        """
        Args:
            data_dir: base data directory
            split: train/val/test
            transform: callable that takes (noisy, clean) numpy arrays and returns transformed arrays
            load_auxiliary: whether to load auxiliary images
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.load_auxiliary = load_auxiliary

        # 支持两种可能的目录命名
        candidate_noisy_dirs = [
            os.path.join(data_dir, split, "noisy_images"),
            os.path.join(data_dir, split, "noisy"),
        ]
        candidate_clean_dirs = [
            os.path.join(data_dir, split, "clean_images"),
            os.path.join(data_dir, split, "clean"),
        ]

        self.noisy_dir = next((d for d in candidate_noisy_dirs if os.path.exists(d)), candidate_noisy_dirs[0])
        self.clean_dir = next((d for d in candidate_clean_dirs if os.path.exists(d)), candidate_clean_dirs[0])
        self.aux_dir = os.path.join(data_dir, split, "auxiliary") if load_auxiliary else None

        # 文件类型过滤
        def list_images(path):
                t_clean = torch.from_numpy(sample['clean'].astype(np.float32))
                item = {'noisy': t_noisy, 'clean': t_clean, 'filename': sample['filename']}
                if 'auxiliary' in sample:
                    item['auxiliary'] = torch.from_numpy(sample['auxiliary'].astype(np.float32))
                return item
        return DataLoader(_TorchDataset(dataset), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    elif backend == 'jittor':
        # For jittor, the top-level script will set the dataset attrs (if jittor is installed)
        return dataset
    else:
        # Python simple generator
        def _generator():
            idxs = list(range(len(dataset)))
            if shuffle:
                import random
                random.shuffle(idxs)
            batch = []
            for i in idxs:
                batch.append(dataset[i])
                if len(batch) >= batch_size:
                    # collate
                    noisy = np.stack([b['noisy'] for b in batch], axis=0)
                    clean = np.stack([b['clean'] for b in batch], axis=0)
                    filenames = [b['filename'] for b in batch]
                    item = {'noisy': noisy, 'clean': clean, 'filename': filenames}
                    yield item
                    batch = []
            if batch:
                noisy = np.stack([b['noisy'] for b in batch], axis=0)
                clean = np.stack([b['clean'] for b in batch], axis=0)
                filenames = [b['filename'] for b in batch]
                yield {'noisy': noisy, 'clean': clean, 'filename': filenames}
        return _generator()

    # The remainder of this file contained older Jittor-specific dataset definitions
    # and duplicates which could cause importing jittor at module import time.
    # We intentionally end the module here to keep it backend-agnostic and
    # compatible with both torch and jittor backends (conversion is handled in
    # the training script).


"""
蒙特卡洛渲染去噪数据集
"""
import os
import jittor as jt
from jittor.dataset import Dataset
import numpy as np
from PIL import Image, ImageFile

# 允许加载截断的 PNG/JPEG，防止报 OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MonteCarloDenoiseDataset(Dataset):
    """
    数据结构：
    data_dir/
      train|val|test/
        noisy_images/
        clean_images/
        auxiliary/   (可选)
    """
    def __init__(self, data_dir, split="train", transform=None, load_auxiliary=False):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.load_auxiliary = load_auxiliary

        self.noisy_dir = os.path.join(data_dir, split, "noisy_images")
        self.clean_dir = os.path.join(data_dir, split, "clean_images")
        self.aux_dir = os.path.join(data_dir, split, "auxiliary") if load_auxiliary else None

        self.noisy_files = sorted([f for f in os.listdir(self.noisy_dir)]) if os.path.exists(self.noisy_dir) else []
        self.clean_files = sorted([f for f in os.listdir(self.clean_dir)]) if os.path.exists(self.clean_dir) else []

        if len(self.noisy_files) != len(self.clean_files):
            common = sorted(list(set(self.noisy_files) & set(self.clean_files)))
            self.noisy_files = common
            self.clean_files = common

        self.total_len = len(self.noisy_files)
        print(f"Loaded {self.total_len} pairs for {split}")

    def __len__(self):
        return self.total_len

    def load_image(self, path):
        if path.lower().endswith(".exr"):
            try:
                import imageio
                img = imageio.imread(path)
            except Exception:
                img = np.zeros((256, 256, 3), dtype=np.float32)
        else:
            img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        if img.ndim == 3:
            img = img.transpose(2, 0, 1)
        else:
            img = img[None, ...]
        return img

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])

        noisy = jt.array(self.load_image(noisy_path))
        clean = jt.array(self.load_image(clean_path))

        if self.transform:
            noisy, clean = self.transform(noisy, clean)

        sample = {"noisy": noisy, "clean": clean, "filename": self.noisy_files[idx]}

        if self.load_auxiliary and self.aux_dir:
            aux_path = os.path.join(self.aux_dir, self.noisy_files[idx])
            if os.path.exists(aux_path):
                sample["auxiliary"] = jt.array(self.load_image(aux_path))
        return sample


def create_data_loader(dataset, batch_size=4, shuffle=True, num_workers=4):
    dataset.set_attrs(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataset
"""
蒙特卡洛渲染去噪数据集
"""
import os
import jittor as jt
from jittor.dataset import Dataset
import numpy as np
from PIL import Image


class MonteCarloDenoiseDataset(Dataset):
    """
    蒙特卡洛渲染去噪数据集

    数据集结构：
    - noisy_images/: 含噪图像（低采样数）
    - clean_images/: 干净图像（高采样数或参考图像）
    - auxiliary/: 辅助信息（可选，如法线、深度等）

    Args:
        data_dir: 数据目录路径
        split: 'train', 'val', 'test'
        transform: 数据增强函数
        load_auxiliary: 是否加载辅助信息
    """

    def __init__(self, data_dir, split='train', transform=None, load_auxiliary=False):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.load_auxiliary = load_auxiliary

        # 构建路径
        self.noisy_dir = os.path.join(data_dir, split, 'noisy_images')
        self.clean_dir = os.path.join(data_dir, split, 'clean_images')
        self.aux_dir = os.path.join(data_dir, split, 'auxiliary') if load_auxiliary else None

        # 获取文件列表
        if os.path.exists(self.noisy_dir):
            self.noisy_files = sorted([f for f in os.listdir(self.noisy_dir)
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.exr'))])
        else:
            self.noisy_files = []

        if os.path.exists(self.clean_dir):
            self.clean_files = sorted([f for f in os.listdir(self.clean_dir)
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.exr'))])
        else:
            self.clean_files = []

        # 确保 noisy 和 clean 文件数量一致
        if len(self.noisy_files) != len(self.clean_files):
            print(f"Warning: Mismatch in file counts. Noisy: {len(self.noisy_files)}, Clean: {len(self.clean_files)}")
            noisy_set = set(self.noisy_files)
            clean_set = set(self.clean_files)
            common_files = sorted(list(noisy_set & clean_set))
            self.noisy_files = common_files
            self.clean_files = common_files

        self.total_len = len(self.noisy_files)
        print(f"Loaded {self.total_len} image pairs for {split} split")

    def __len__(self):
        return self.total_len

    def load_image(self, path):
        """加载图像，支持 PNG/JPG 和 EXR 格式"""
        if path.lower().endswith('.exr'):
            try:
                import imageio
                img = imageio.imread(path)
            except Exception:
                print(f"Warning: Cannot load EXR file {path}, using placeholder")
                img = np.zeros((256, 256, 3), dtype=np.float32)
        else:
            try:
                img = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
            except Exception as e:
                print(f"Warning: cannot read image {path}: {e}; using zeros")
                img = np.zeros((256, 256, 3), dtype=np.float32)

        # 转换为 CHW
        if len(img.shape) == 3:
            img = img.transpose(2, 0, 1)
        else:
            img = img[np.newaxis, :, :]
        return img

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])

        noisy_img = self.load_image(noisy_path)
        clean_img = self.load_image(clean_path)

        noisy_img = jt.array(noisy_img)
        clean_img = jt.array(clean_img)

        if self.transform:
            noisy_img, clean_img = self.transform(noisy_img, clean_img)

        result = {
            'noisy': noisy_img,
            'clean': clean_img,
            'filename': self.noisy_files[idx]
        }

        if self.load_auxiliary and self.aux_dir:
            aux_path = os.path.join(self.aux_dir, self.noisy_files[idx])
            if os.path.exists(aux_path):
                aux_img = self.load_image(aux_path)
                result['auxiliary'] = jt.array(aux_img)

        return result


def create_data_loader(dataset, batch_size=4, shuffle=True, num_workers=4):
    """创建数据加载器"""
    dataset.set_attrs(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataset


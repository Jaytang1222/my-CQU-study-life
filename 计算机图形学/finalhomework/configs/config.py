"""
基础配置
"""
import os


class Config:
    # 数据
    data_dir = "./dataset"
    batch_size = 4
    num_workers = 4
    crop_size = 256
    use_augmentation = True
    load_auxiliary = False
    # 多尺度裁剪
    enable_multi_scale = True
    multi_scales = [256, 320, 384]
    
    # Patch-based训练（用于大尺寸图像）
    use_patch_training = False  # 是否使用patch-based训练
    patch_size = 512  # patch大小
    patches_per_image = 4  # 每张图提取的patch数量
    overlap = 64  # patch之间的重叠像素

    # 模型
    n_channels = 3
    n_classes = 3
    bilinear = True
    use_attention = True  # 使用注意力机制
    use_residual_learning = True  # 使用残差学习（学习噪声而不是直接学习干净图像）

    # 训练
    num_epochs = 150  # 增加训练轮数，让模型充分学习
    learning_rate = 1e-4
    weight_decay = 1e-5
    optimizer = "adam"
    scheduler = "cosine"
    scheduler_params = {
        "cosine": {"T_max": 100, "eta_min": 1e-6},
        "step": {"step_size": 30, "gamma": 0.1},
    }
    loss_type = "combined"  # l1 / l2 / combined / ssim / perceptual
    loss_weights = {
        "l1": 0.2,         # L1损失（增加，更强调去噪）
        "l2": 0.2,         # L2损失（增加，更强调去噪）
        "ssim": 0.25,      # SSIM损失（增加，保持结构）
        "smooth": 0.02,    # 平滑损失（大幅减少，避免过度平滑导致模糊）
        "color": 0.08,     # Lab颜色空间损失（减小色差）
        "chroma": 0.05,    # 色度损失（只关注颜色，忽略亮度）
        "grad": 0.15,      # 梯度损失（增加，更强调保持边缘和细节）
        "edge": 0.02,      # 边缘保持损失（减少，避免过度平滑）
        "perceptual": 0.1, # 感知损失（多尺度特征匹配，减少结构失真）
        "freq": 0.05,      # 频率域损失（减少伪影和失真）
        "multiscale": 0.08 # 多尺度损失（增加，在不同尺度上保持一致性）
    }
    seed = 42

    # 保存与日志
    save_dir = "./checkpoints"
    log_dir = "./logs"
    save_freq = 10
    val_freq = 1

    # 设备
    use_gpu = True
    gpu_id = 0

    resume_from = None

    # Early stopping
    early_stop_metric = "psnr"  # 可选: psnr / val_loss
    early_stop_patience = 15    # 连续无提升的 epoch 数（增加，给模型更多时间学习）

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: Unknown config key: {k}")

    def print_config(self):
        print("=" * 50)
        print("Config:")
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                print(f"{k}: {v}")
        print("=" * 50)


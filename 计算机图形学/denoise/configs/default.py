config = {
    # 训练/验证数据路径
    "clean_root": "dataset/train/clean",
    "noisy_root": "dataset/train/noisy",
    "val_clean_root": "dataset/val/clean",
    "val_noisy_root": "dataset/val/noisy",

    # 图像与训练参数
    "image_size": 256,
    "batch_size": 4,
    "num_workers": 0,
    "learning_rate": 1e-3,
    "num_epochs": 20,
    "save_dir": "checkpoints",
    "log_interval": 10,
    "val_interval": 1,

    # 采样通道信息
    "num_mc_samples": 1,
    "max_mc_samples": 1,
    "noise_sigma": 0.0,  
    # 模型与随机种子
    "seed": 42,
    "base_channels": 32,
}


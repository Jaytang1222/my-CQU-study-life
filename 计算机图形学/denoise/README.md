## Jittor Monte Carlo 图像降噪（CPU）

一个基于 Jittor 的简易蒙特卡洛图像降噪示例，仅使用 CPU 训练。核心组件按模块拆分：配置、数据集、模型、损失与训练脚本，便于维护与扩展。模型采用轻量残差 CNN（无上下采样），适合 CPU。数据集采用成对 clean/noisy 图片。

### 目录结构
- `configs/`: 训练配置
- `data/`: 数据集定义（成对 clean/noisy）
- `losses/`: 损失与评估指标
- `models/`: 网络结构（默认使用 `ResidualDenoiser`）
- `utils/`: 工具（配置加载、检查点）
- `train.py`: 训练入口

### 快速开始
1) 安装依赖（需 CPU 版 Jittor）
```
python -m pip install jittor pillow numpy
```
若机器同时有 GPU，请强制 CPU：
```
import jittor as jt
jt.flags.use_cuda = 0
```

2) 准备数据（成对 clean/noisy，同名文件配对）  
示例：  
```
data/train/clean/xxx.png
data/train/noisy/xxx.png
data/val/clean/yyy.png   # 可选
data/val/noisy/yyy.png   # 可选
```

3) 运行训练
```
python train.py --config configs/default.py \
  --clean_root data/train/clean --noisy_root data/train/noisy \
  --val_clean_root data/val/clean --val_noisy_root data/val/noisy
```
若无验证集可省略 val 参数。可覆盖：`--num_epochs`、`--batch_size`、`--noise_sigma` 等。

### 配置字段示例
- `clean_root` / `noisy_root`: 训练 clean/noisy 目录
- `val_clean_root` / `val_noisy_root`: 验证 clean/noisy 目录（可为空）
- `num_mc_samples` / `max_mc_samples`: 样本数通道归一化；真实 noisy 一般设为 1/1
- `noise_sigma`: 可选，对 noisy 再叠加微噪声做数据增强
- `base_channels`: 残差 CNN 基础通道数（越大越准但越耗时）

### 说明
- 模型输入为 `4` 通道：`[noisy_rgb(3), sample_count_map(1)]`，网络输出干净 RGB。
- 训练与验证同时报告 L1 损失与 PSNR。
- 仅使用 CPU，适合小规模演示或原型验证。
- 主要模型：`models/resnet_denoiser.py`（3x3 卷积 + 残差块），无需上下采样，CPU 友好。


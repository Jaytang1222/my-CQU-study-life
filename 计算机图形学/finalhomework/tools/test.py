"""
可视化对比脚本：
对 test 集逐样本输出三张拼接图：原始 noisy、GT、模型降噪结果。

用法示例：
python tools/visualize.py \
  --checkpoint ./checkpoints/best_model.pkl \
  --data_dir /home/students/undergraduate/zhengly/workspace/denoise/dataset \
  --save_dir ./viz \
  --max_samples 20 \
  --gpu_id 0
"""
import os
import sys
import argparse
import jittor as jt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# 添加项目根目录到路径，以便导入模块
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import UNet
from dataset import MonteCarloDenoiseDataset, create_data_loader
from configs.config import Config
from utils.transforms import get_val_transforms
from utils.metrics import compute_metrics, print_metrics


def to_uint8(img_np):
    """输入 [C,H,W] 或 [H,W,C]，值可能在 [0,1] 或 [0,255]，输出 uint8 HWC"""
    if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
        img_np = img_np.transpose(1, 2, 0)
    
    # 检查值域：如果最大值>1，假设已经是[0,255]范围
    img_max = img_np.max()
    img_min = img_np.min()
    
    if img_max > 1.0:
        # 已经是[0,255]范围，直接clip并转换
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    else:
        # [0,1]范围，转换为[0,255]
        img_np = np.clip(img_np, 0, 1) * 255.0
        img_np = img_np.astype(np.uint8)
    
    return img_np


def add_text_below(img_array, text):
    """在图像下方添加文字标签，返回带标题的完整图像"""
    img = Image.fromarray(img_array)
    
    # 尝试加载字体
    font = None
    font_size = max(24, img.height // 12)
    
    # 尝试多个常见字体路径
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Windows/Fonts/arial.ttf",
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        except:
            continue
    
    # 如果都失败，使用默认字体
    if font is None:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # 创建临时图像来计算文字尺寸
    temp_img = Image.new('RGB', (100, 100))
    temp_draw = ImageDraw.Draw(temp_img)
    
    if font:
        try:
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            text_width = len(text) * 10
            text_height = 20
    else:
        text_width = len(text) * 10
        text_height = 20
    
    # 标题区域高度（文字高度 + 上下padding）
    title_height = text_height + 20
    
    # 创建新图像：原图高度 + 标题高度
    new_height = img.height + title_height
    new_img = Image.new('RGB', (img.width, new_height), color=(255, 255, 255))
    
    # 将原图粘贴到顶部
    new_img.paste(img, (0, 0))
    
    # 绘制标题
    draw = ImageDraw.Draw(new_img)
    x = (img.width - text_width) // 2
    y = img.height + 10  # 原图下方10像素开始
    
    # 绘制文字（黑色）
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    
    return np.array(new_img)


def apply_postprocess(img_uint8, mode="none", median_radius=1, unsharp_radius=1.5, unsharp_amount=0.3):
    """对 uint8 HWC 图像做轻量后处理：中值滤波 + 锐化"""
    if mode == "none":
        return img_uint8

    img = Image.fromarray(img_uint8)

    if "median" in mode:
        # size 必须为奇数
        ksize = max(3, 2 * median_radius + 1)
        img = img.filter(ImageFilter.MedianFilter(size=ksize))

    if "unsharp" in mode:
        blur = img.filter(ImageFilter.GaussianBlur(radius=unsharp_radius))
        img_np = np.array(img, dtype=np.float32)
        blur_np = np.array(blur, dtype=np.float32)
        sharp = img_np + unsharp_amount * (img_np - blur_np)
        img_uint8 = np.clip(sharp, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_uint8)

    return np.array(img, dtype=np.uint8)


def patch_based_inference(model, noisy_img, patch_size=512, overlap=64):
    """
    对大图进行patch-based推理，返回原图尺寸的降噪结果
    Args:
        model: 训练好的模型
        noisy_img: jittor tensor [C, H, W]，输入图像
        patch_size: patch大小
        overlap: patch之间的重叠像素
    Returns:
        pred: jittor tensor [C, H, W]，原图尺寸的降噪结果
    """
    import jittor as jt
    C, H, W = noisy_img.shape
    
    # 如果图像小于patch_size，直接推理
    if H <= patch_size and W <= patch_size:
        noisy_batch = noisy_img.unsqueeze(0)  # [1, C, H, W]
        pred_batch = model(noisy_batch)
        return pred_batch[0]  # [C, H, W]
    
    # 计算步长
    step = patch_size - overlap
    
    # 计算需要的patch数量
    h_patches = (H - overlap + step - 1) // step
    w_patches = (W - overlap + step - 1) // step
    
    # 创建输出图像
    pred_full = jt.zeros((C, H, W), dtype=noisy_img.dtype)
    count_map = jt.zeros((H, W), dtype=jt.float32)
    
    # 对每个patch进行推理
    for hi in range(h_patches):
        for wi in range(w_patches):
            # 计算patch的坐标
            y_start = min(hi * step, H - patch_size)
            y_end = y_start + patch_size
            x_start = min(wi * step, W - patch_size)
            x_end = x_start + patch_size
            
            # 提取patch
            patch = noisy_img[:, y_start:y_end, x_start:x_end]  # [C, patch_size, patch_size]
            patch_batch = patch.unsqueeze(0)  # [1, C, patch_size, patch_size]
            
            # 推理
            pred_patch = model(patch_batch)[0]  # [C, patch_size, patch_size]
            
            # 累加到输出图像（重叠区域会累加多次）
            pred_full[:, y_start:y_end, x_start:x_end] += pred_patch
            count_map[y_start:y_end, x_start:x_end] += 1.0
    
    # 平均重叠区域
    count_map = jt.maximum(count_map, 1.0)  # 避免除零
    pred_full = pred_full / count_map.unsqueeze(0)  # [C, H, W]
    
    return pred_full


def concat_triplet(noisy, clean, pred):
    """垂直拼接三张 HWC 图，并在每张图下方添加标题"""
    # 在每张图下方添加标题
    noisy_labeled = add_text_below(noisy, "Noisy (Input)")
    pred_labeled = add_text_below(pred, "Denoised (Predicted)")
    clean_labeled = add_text_below(clean, "GT (Ground Truth)")
    
    # 确保三张图宽度一致（取最大宽度）
    max_width = max(noisy_labeled.shape[1], pred_labeled.shape[1], clean_labeled.shape[1])
    
    # 如果宽度不一致，进行padding
    def pad_to_width(img_arr, target_width):
        if img_arr.shape[1] < target_width:
            pad_width = target_width - img_arr.shape[1]
            pad = np.zeros((img_arr.shape[0], pad_width, img_arr.shape[2]), dtype=img_arr.dtype)
            pad.fill(255)  # 白色填充
            return np.concatenate([img_arr, pad], axis=1)
        return img_arr
    
    noisy_labeled = pad_to_width(noisy_labeled, max_width)
    pred_labeled = pad_to_width(pred_labeled, max_width)
    clean_labeled = pad_to_width(clean_labeled, max_width)
    
    # 垂直拼接（从上到下）
    return np.concatenate([noisy_labeled, pred_labeled, clean_labeled], axis=0)


def main():
    ap = argparse.ArgumentParser(description="Visualize denoising results on test set")
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    ap.add_argument("--data_dir", type=str, default=None, help="Test data directory")
    ap.add_argument("--save_dir", type=str, default="./viz", help="Output directory")
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    ap.add_argument("--max_samples", type=int, default=None, help="Max samples to save (None for all)")
    ap.add_argument("--gpu_id", type=int, default=None, help="GPU id")
    ap.add_argument("--resize", type=int, default=None, 
                    help="Resize images to this size (None=auto-detect, 0=keep original, may cause OOM for large images)")
    ap.add_argument("--use_patch_inference", action="store_true",
                    help="Use patch-based inference for large images (process in patches and merge back to original size)")
    ap.add_argument("--patch_size", type=int, default=512, help="Patch size for patch-based inference")
    ap.add_argument("--patch_overlap", type=int, default=64, help="Overlap between patches")
    ap.add_argument("--postprocess", choices=["none", "median", "unsharp", "median_unsharp"], default="none",
                    help="轻量后处理：中值滤波/反锐化避免残留噪点或模糊")
    ap.add_argument("--median_radius", type=int, default=1, help="中值滤波半径（像素）")
    ap.add_argument("--unsharp_radius", type=float, default=1.5, help="GaussianBlur半径（锐化用）")
    ap.add_argument("--unsharp_amount", type=float, default=0.35, help="锐化强度，0-1之间更安全")
    args = ap.parse_args()

    config = Config()
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.gpu_id is not None:
        config.gpu_id = args.gpu_id

    # 设备
    if jt.has_cuda:
        if args.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        jt.flags.use_cuda = 1
        print(f"Using GPU {config.gpu_id}")
    else:
        jt.flags.use_cuda = 0
        print("Using CPU")

    # 模型
    model = UNet(
        config.n_channels, 
        config.n_classes, 
        config.bilinear,
        use_attention=getattr(config, 'use_attention', True),
        use_residual_learning=getattr(config, 'use_residual_learning', True)
    )
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = jt.load(args.checkpoint)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint epoch {ckpt.get('epoch', '?')}")

    # 数据（推理时使用单进程，避免多进程问题）
    # 对于大尺寸图像（>2000px），建议使用resize避免GPU内存问题
    # 如果原图很大且不resize，会导致GPU内存不足
    if args.use_patch_inference:
        # Patch-based推理：保持原图尺寸，分块处理
        val_transform = None  # 不resize，保持原图尺寸
        print(f"Using patch-based inference (patch_size={args.patch_size}, overlap={args.patch_overlap})")
        print("  Images will be processed in patches and merged back to original size")
    elif args.resize == 0:
        # 明确禁用resize，使用原图尺寸（可能OOM）
        val_transform = None
        print("Keeping original image size (--resize 0 specified, may cause OOM for large images)")
    elif args.resize:
        val_transform = get_val_transforms(args.resize)
        print(f"Resizing images to {args.resize}x{args.resize}")
    else:
        # 检查是否需要自动resize（如果图像太大）
        # 先加载一张图检查尺寸
        test_ds_temp = MonteCarloDenoiseDataset(config.data_dir, "test", transform=None, load_auxiliary=config.load_auxiliary)
        if len(test_ds_temp) > 0:
            sample = test_ds_temp[0]
            h, w = sample['noisy'].shape[1], sample['noisy'].shape[2]
            max_dim = max(h, w)
            if max_dim > 2000:
                # 自动resize到1024，避免内存问题
                auto_size = 1024
                val_transform = get_val_transforms(auto_size)
                print(f"Large images detected ({w}x{h}), auto-resizing to {auto_size}x{auto_size} to avoid GPU memory issues")
                print(f"  (Use --resize to specify custom size, --resize 0 to disable, or --use_patch_inference for patch-based inference)")
            else:
                val_transform = None
                print(f"Keeping original image size ({w}x{h})")
        else:
            val_transform = None
            print("Keeping original image size (may cause memory issues for large images)")
    
    test_ds = MonteCarloDenoiseDataset(config.data_dir, "test", transform=val_transform, load_auxiliary=config.load_auxiliary)
    # 使用batch_size=1避免大图像导致的内存问题
    test_loader = create_data_loader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    os.makedirs(args.save_dir, exist_ok=True)
    count = 0
    print(f"Starting inference on {len(test_ds)} samples...")
    if args.postprocess != "none":
        print(f"[Postprocess] mode={args.postprocess}, median_radius={args.median_radius}, "
              f"unsharp_radius={args.unsharp_radius}, unsharp_amount={args.unsharp_amount}")
    print("=" * 80)

    # 用于累计评估指标
    total_mse_sum = 0.0
    total_mae_sum = 0.0
    total_pixels = 0
    total_ssim_sum = 0.0
    sample_metrics = []

    with jt.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            print(f"Processing batch {batch_idx + 1}...")
            noisy = batch["noisy"]
            clean = batch["clean"]
            names = batch["filename"]
            if isinstance(names, str):
                names = [names] * noisy.shape[0]

            if args.use_patch_inference:
                # Patch-based推理：对每张图分别处理
                pred_list = []
                for i in range(noisy.shape[0]):
                    noisy_single = noisy[i]  # [C, H, W]
                    pred_single = patch_based_inference(
                        model, noisy_single, 
                        patch_size=args.patch_size, 
                        overlap=args.patch_overlap
                    )
                    pred_list.append(pred_single)
                pred = jt.stack(pred_list, dim=0)  # [B, C, H, W]
            else:
                # 常规推理
                pred = model(noisy)
            
            pred = jt.clamp(pred, 0, 1)
            # 同步GPU操作
            jt.sync_all()

            for i in range(pred.shape[0]):
                # 获取图像尺寸信息
                processed_h, processed_w = noisy[i].shape[1], noisy[i].shape[2]
                
                # 尝试读取原始图像尺寸
                filename = os.path.basename(names[i])
                original_size = None
                if config.data_dir:
                    noisy_path = os.path.join(config.data_dir, "test", "noisy", names[i])
                    if os.path.exists(noisy_path):
                        try:
                            with Image.open(noisy_path) as img:
                                original_size = img.size  # (width, height)
                        except:
                            pass
                
                # 计算评估指标（使用clamp后的pred）
                pred_eval = pred[i:i+1]  # 保持batch维度
                clean_eval = clean[i:i+1]
                metrics = compute_metrics(pred_eval, clean_eval, max_val=1.0)
                
                # 累计指标（按像素数加权）
                batch_pixels = pred_eval.numel()
                total_mse_sum += metrics["mse"] * batch_pixels
                total_mae_sum += metrics["mae"] * batch_pixels
                total_pixels += batch_pixels
                total_ssim_sum += metrics["ssim"]
                sample_metrics.append(metrics)
                
                # 输出单张图片的指标和尺寸信息
                print(f"\n[{count + 1}] {filename}:")
                if original_size:
                    print(f"  Original size: {original_size[0]}x{original_size[1]}")
                    print(f"  Processed size: {processed_w}x{processed_h}")
                    if original_size[0] != processed_w or original_size[1] != processed_h:
                        print(f"  (Resized from {original_size[0]}x{original_size[1]} to {processed_w}x{processed_h})")
                else:
                    print(f"  Image size: {processed_w}x{processed_h}")
                print(f"  MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}, "
                      f"PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
                
                # 保存可视化结果
                # 先同步并转换为numpy
                noisy_np = noisy[i].numpy()
                clean_np = clean[i].numpy()
                pred_np = pred[i].numpy()
                
                # 调试：输出值域信息
                if count == 0:  # 只在第一张图时输出
                    print(f"  [Debug] noisy range: [{noisy_np.min():.4f}, {noisy_np.max():.4f}], mean: {noisy_np.mean():.4f}")
                    print(f"  [Debug] clean range: [{clean_np.min():.4f}, {clean_np.max():.4f}], mean: {clean_np.mean():.4f}")
                    print(f"  [Debug] pred range: [{pred_np.min():.4f}, {pred_np.max():.4f}], mean: {pred_np.mean():.4f}")
                
                noisy_i = to_uint8(noisy_np)
                clean_i = to_uint8(clean_np)
                pred_i = to_uint8(pred_np)
                # 后处理：中值滤波 + 锐化（可选）
                pred_i = apply_postprocess(
                    pred_i,
                    mode=args.postprocess,
                    median_radius=args.median_radius,
                    unsharp_radius=args.unsharp_radius,
                    unsharp_amount=args.unsharp_amount,
                )
                
                # 验证转换后的值域
                if count == 0:
                    print(f"  [Debug] noisy_i uint8 range: [{noisy_i.min()}, {noisy_i.max()}], mean: {noisy_i.mean():.1f}")
                    print(f"  [Debug] clean_i uint8 range: [{clean_i.min()}, {clean_i.max()}], mean: {clean_i.mean():.1f}")
                    print(f"  [Debug] pred_i uint8 range: [{pred_i.min()}, {pred_i.max()}], mean: {pred_i.mean():.1f}")
                
                merged = concat_triplet(noisy_i, clean_i, pred_i)
                
                # 验证merged图像
                if merged.min() == merged.max() == 0:
                    print(f"  [Warning] Merged image is all zeros!")
                elif merged.max() < 10:
                    print(f"  [Warning] Merged image is very dark (max={merged.max()})")
                
                base = os.path.splitext(names[i])[0]
                out_path = os.path.join(args.save_dir, f"{base}_compare.png")
                
                # 确保图像格式正确
                if merged.dtype != np.uint8:
                    merged = merged.astype(np.uint8)
                if merged.shape[2] != 3:
                    print(f"  [Warning] Unexpected image shape: {merged.shape}")
                
                Image.fromarray(merged, mode='RGB').save(out_path)
                count += 1
                print(f"  Saved: {out_path}")
                
                if args.max_samples is not None and count >= args.max_samples:
                    break
            
            if args.max_samples is not None and count >= args.max_samples:
                break

    # 计算整体平均指标
    print("\n" + "=" * 80)
    print("Overall Evaluation Results:")
    print("=" * 80)
    if total_pixels > 0:
        overall_mse = total_mse_sum / total_pixels
        overall_mae = total_mae_sum / total_pixels
        overall_ssim = total_ssim_sum / count if count > 0 else 0.0
        if overall_mse > 0:
            overall_psnr = 20 * np.log10(1.0 / np.sqrt(overall_mse))
        else:
            overall_psnr = float("inf")
        
        overall_metrics = {
            "mse": overall_mse,
            "mae": overall_mae,
            "psnr": overall_psnr,
            "ssim": overall_ssim
        }
        print_metrics(overall_metrics, prefix="Test")
        print(f"\nTotal samples evaluated: {count}")
    else:
        print("No samples were evaluated.")
    
    print(f"\nSaved {count} visualization images to {args.save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()


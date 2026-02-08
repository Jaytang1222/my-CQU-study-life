"""
训练脚本（支持多尺度裁剪、可配置损失、early stop）
"""
import os
import math
import argparse
import importlib
has_jittor = False
has_torch = False
try:
    import jittor as jt
    from jittor import optim as jt_optim
    has_jittor = True
except Exception:
    has_jittor = False

try:
    import torch
    import torch.optim as torch_optim
    has_torch = True
except Exception:
    has_torch = False

from models import get_unet
from dataset import MonteCarloDenoiseDataset, create_data_loader
from utils.transforms import get_train_transforms, get_val_transforms
if has_jittor:
    from utils.losses import get_loss_function as get_loss_function_jt
    from utils.metrics import print_metrics as print_metrics_jt
else:
    get_loss_function_jt = None
    print_metrics_jt = None

if has_torch:
    from utils.losses_torch import get_loss_function as get_loss_function_torch
    from utils.metrics_torch import print_metrics as print_metrics_torch
else:
    get_loss_function_torch = None
    print_metrics_torch = None
from configs.config import Config


def train_epoch_jittor(model, loader, criterion, optimizer, epoch, config, total_batches):
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(loader):
        noisy, clean = batch["noisy"], batch["clean"]
        pred = model(noisy)
        pred_eval = jt.clamp(pred, 0, 1)
        loss = criterion(pred_eval, clean)
        optimizer.step(loss)
        total_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"[Epoch {epoch}] Batch {i+1}/{total_batches} Loss {loss.item():.6f}")
    return total_loss / max(1, total_batches)


def train_epoch_torch(model, loader, criterion, optimizer, epoch, config, total_batches, device):
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(loader):
        noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
        optimizer.zero_grad()
        pred = model(noisy)
        pred_eval = torch.clamp(pred, 0, 1)
        loss = criterion(pred_eval, clean)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"[Epoch {epoch}] Batch {i+1}/{total_batches} Loss {loss.item():.6f}")
    return total_loss / max(1, total_batches)


def validate_jittor(model, loader, criterion):
    model.eval()
    tot_loss = 0.0
    # 累加所有batch的误差平方和、绝对误差和、像素总数，用于正确计算整体指标
    total_mse_sum = 0.0
    total_mae_sum = 0.0
    total_pixels = 0
    total_ssim_sum = 0.0
    batch_count = 0
    min_max_seen = []
    extreme_count = 0
    import numpy as np
    with jt.no_grad():
        for batch in loader:
            noisy, clean = batch["noisy"], batch["clean"]
            pred = model(noisy)
            pred_eval = jt.clamp(pred, 0, 1)
            loss = criterion(pred_eval, clean)
            tot_loss += loss.item()
            
            # 累加MSE和MAE（按像素数加权）
            batch_mse = jt.mean((pred_eval - clean) ** 2).item()
            batch_mae = jt.mean(jt.abs(pred_eval - clean)).item()
            batch_pixels = pred_eval.numel()
            total_mse_sum += batch_mse * batch_pixels
            total_mae_sum += batch_mae * batch_pixels
            total_pixels += batch_pixels
            
            # SSIM可以平均（因为是归一化的）
            from utils.metrics import ssim
            total_ssim_sum += ssim(pred_eval, clean)
            batch_count += 1
            
            if len(min_max_seen) < 3:
                min_max_seen.append((float(pred.min()), float(pred.max())))
            # 统计极端样本：任一维度超出 [-0.1, 1.1] 视为极值
            if float(pred.min()) < -0.1 or float(pred.max()) > 1.1:
                extreme_count += 1
    
    # 计算整体指标
    agg = {}
    if total_pixels > 0:
        agg["mse"] = total_mse_sum / total_pixels
        agg["mae"] = total_mae_sum / total_pixels
        # 从整体MSE计算PSNR
        if agg["mse"] > 0:
            agg["psnr"] = 20 * np.log10(1.0 / np.sqrt(agg["mse"]))
        else:
            agg["psnr"] = float("inf")
        agg["ssim"] = total_ssim_sum / batch_count if batch_count > 0 else 0.0
    else:
        agg = {"mse": 0.0, "mae": 0.0, "psnr": 0.0, "ssim": 0.0}
    
    if len(loader) > 0:
        tot_loss /= len(loader)
    
    if min_max_seen:
        mins = [p[0] for p in min_max_seen]
        maxs = [p[1] for p in min_max_seen]
        print(f"[Val clamp log] pred min range: {min(mins):.4f}~{max(mins):.4f}, "
              f"pred max range: {min(maxs):.4f}~{max(maxs):.4f}")
    if extreme_count > 0:
        print(f"[Val clamp log] extreme samples (pred outside [-0.1,1.1]): {extreme_count}")
    return tot_loss, agg


def validate_torch(model, loader, criterion, device):
    model.eval()
    tot_loss = 0.0
    total_mse_sum = 0.0
    total_mae_sum = 0.0
    total_pixels = 0
    total_ssim_sum = 0.0
    batch_count = 0
    min_max_seen = []
    extreme_count = 0
    import numpy as np
    with torch.no_grad():
        for batch in loader:
            noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
            pred = model(noisy)
            pred_eval = torch.clamp(pred, 0, 1)
            loss = criterion(pred_eval, clean)
            tot_loss += loss.item()

            batch_mse = float(torch.mean((pred_eval - clean) ** 2).item())
            batch_mae = float(torch.mean(torch.abs(pred_eval - clean)).item())
            batch_pixels = pred_eval.numel()
            total_mse_sum += batch_mse * batch_pixels
            total_mae_sum += batch_mae * batch_pixels
            total_pixels += batch_pixels

            # SSIM fallback using numpy
            from utils.metrics_torch import ssim
            total_ssim_sum += ssim(pred_eval, clean)
            batch_count += 1

            if len(min_max_seen) < 3:
                min_max_seen.append((float(pred.min()), float(pred.max())))
            if float(pred.min()) < -0.1 or float(pred.max()) > 1.1:
                extreme_count += 1

    agg = {}
    if total_pixels > 0:
        agg["mse"] = total_mse_sum / total_pixels
        agg["mae"] = total_mae_sum / total_pixels
        if agg["mse"] > 0:
            agg["psnr"] = 20 * np.log10(1.0 / np.sqrt(agg["mse"]))
        else:
            agg["psnr"] = float('inf')
        agg["ssim"] = total_ssim_sum / batch_count if batch_count > 0 else 0.0
    else:
        agg = {"mse": 0.0, "mae": 0.0, "psnr": 0.0, "ssim": 0.0}

    if len(loader) > 0:
        tot_loss /= len(loader)

    if min_max_seen:
        mins = [p[0] for p in min_max_seen]
        maxs = [p[1] for p in min_max_seen]
        print(f"[Val clamp log] pred min range: {min(mins):.4f}~{max(mins):.4f}, "
              f"pred max range: {min(maxs):.4f}~{max(maxs):.4f}")
    if extreme_count > 0:
        print(f"[Val clamp log] extreme samples (pred outside [-0.1,1.1]): {extreme_count}")
    return tot_loss, agg


def main():
    parser = argparse.ArgumentParser(description="Train UNet for Monte Carlo Denoising")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--backend", type=str, default='jittor', choices=['jittor', 'torch'], help='Backend to use: jittor or torch')
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--crop_size", type=int, default=None, help="Crop size for training")
    parser.add_argument("--use_patch", action="store_true", help="Use patch-based training for large images")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size for patch-based training")
    parser.add_argument("--patches_per_image", type=int, default=4, help="Number of patches per image")
    parser.add_argument("--use_precropped", action="store_true", help="Use pre-cropped images (images already cropped into smaller pieces)")
    parser.add_argument("--no_crop", action="store_true", help="Disable RandomCrop augmentation (only resize to fixed size)")
    args = parser.parse_args()

    config = Config()
    if args.data_dir: config.data_dir = args.data_dir
    if args.batch_size: config.batch_size = args.batch_size
    if args.epochs: config.num_epochs = args.epochs
    if args.lr: config.learning_rate = args.lr
    if args.gpu_id is not None: config.gpu_id = args.gpu_id
    if args.resume: config.resume_from = args.resume
    if args.crop_size: config.crop_size = args.crop_size
    if args.use_patch:
        config.use_patch_training = True
        config.patch_size = args.patch_size
        config.patches_per_image = args.patches_per_image
    if args.use_precropped:
        config.use_precropped = True
    use_crop = not args.no_crop  # 如果指定--no_crop，则use_crop=False
    
    # 如果禁用crop，也禁用multi_scale（因为multi_scale需要crop）
    if not use_crop:
        config.enable_multi_scale = False
        print(f"[Info] --no_crop specified, disabling multi_scale augmentation")

    config.print_config()
    backend = args.backend
    if backend == 'jittor' and not has_jittor:
        if has_torch:
            print('[Warn] Jittor not available, falling back to PyTorch backend')
            backend = 'torch'
        else:
            print('[Error] Neither Jittor nor PyTorch is available. Please install one of them.')
            raise SystemExit(1)

    if backend == 'jittor':
        jt.set_global_seed(config.seed)
        if config.use_gpu and jt.has_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
            jt.flags.use_cuda = 1
            print(f"Using GPU {config.gpu_id} for Jittor")
        else:
            jt.flags.use_cuda = 0
            print("Using CPU for Jittor")
    else:
        # torch
        import torch
        torch.manual_seed(config.seed)
        device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        print(f"Torch backend: using device {device}")

    # 根据配置选择训练策略
    use_precropped = getattr(args, 'use_precropped', False) or getattr(config, 'use_precropped', False)
    if use_precropped:
        # 使用预裁剪好的小图：图片已经在数据准备阶段裁剪成小图
        print("Using pre-cropped images (images already cropped into smaller pieces)")
        
        # 检测图片尺寸，自动调整crop_size
        temp_ds = MonteCarloDenoiseDataset(config.data_dir, "train", transform=None, load_auxiliary=config.load_auxiliary)
        if len(temp_ds) > 0:
            sample = temp_ds[0]
            img_h, img_w = sample['noisy'].shape[1], sample['noisy'].shape[2]
            print(f"Detected image size: {img_w}x{img_h}")
            
            # 如果crop_size未指定或大于图片尺寸，自动调整
            if config.crop_size is None or config.crop_size > min(img_h, img_w):
                # 使用图片的最小边作为crop_size，或使用默认值
                auto_crop = min(img_h, img_w)
                if auto_crop > 512:
                    auto_crop = 512  # 限制最大crop_size
                config.crop_size = auto_crop
                print(f"Auto-adjusted crop_size to {config.crop_size} (based on image size)")
        
        # 使用传统训练方式，但crop_size已适配小图
        train_tf = get_train_transforms(
            config.crop_size,
            enable_multi_scale=config.enable_multi_scale,
            multi_scales=config.multi_scales,
            use_crop=use_crop,
        ) if config.use_augmentation else None
        val_tf = get_val_transforms(config.crop_size)
        
        if train_tf is not None:
            print(f"[Transform] Train transforms: crop={use_crop}, multi_scale={config.enable_multi_scale}")
        else:
            print(f"[Transform] No augmentation (use_augmentation=False)")

        train_ds = MonteCarloDenoiseDataset(config.data_dir, "train", transform=train_tf, load_auxiliary=config.load_auxiliary)
        val_ds = MonteCarloDenoiseDataset(config.data_dir, "val", transform=val_tf, load_auxiliary=config.load_auxiliary)
        
        if use_crop:
            print(f"Using crop_size: {config.crop_size}x{config.crop_size} (with RandomCrop)")
        else:
            print(f"Resizing to: {config.crop_size}x{config.crop_size} (no RandomCrop, only ResizeTo)")
        print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        
    elif getattr(config, 'use_patch_training', False):
        # Patch-based训练：不resize，直接从大图提取patch
        print("Using patch-based training for large images")
        train_tf = None  # 不resize，保持原始尺寸
        val_tf = None
        
        # 创建基础数据集（不resize）
        train_base_ds = MonteCarloDenoiseDataset(
            config.data_dir, "train", 
            transform=None,  # 不resize
            load_auxiliary=config.load_auxiliary
        )
        val_base_ds = MonteCarloDenoiseDataset(
            config.data_dir, "val",
            transform=None,  # 不resize
            load_auxiliary=config.load_auxiliary
        )
        
        # 转换为patch数据集
        from utils.patch_loader import create_patch_dataset
        patch_size = getattr(config, 'patch_size', 512)
        patches_per_image = getattr(config, 'patches_per_image', 4)
        overlap = getattr(config, 'overlap', 64)
        
        train_ds = create_patch_dataset(
            train_base_ds, 
            patch_size=patch_size,
            patches_per_image=patches_per_image,
            overlap=overlap
        )
        val_ds = create_patch_dataset(
            val_base_ds,
            patch_size=patch_size,
            patches_per_image=patches_per_image,
            overlap=overlap
        )
        
        print(f"Patch size: {patch_size}x{patch_size}, Patches per image: {patches_per_image}, Overlap: {overlap}")
        print(f"Train patches: {len(train_ds)}, Val patches: {len(val_ds)}")
    else:
        # 传统训练：resize到固定尺寸
        train_tf = get_train_transforms(
            config.crop_size,
            enable_multi_scale=config.enable_multi_scale,
            multi_scales=config.multi_scales,
            use_crop=use_crop,
        ) if config.use_augmentation else None
        val_tf = get_val_transforms(config.crop_size)
        
        if train_tf is not None:
            print(f"[Transform] Train transforms: crop={use_crop}, multi_scale={config.enable_multi_scale}")
        else:
            print(f"[Transform] No augmentation (use_augmentation=False)")

        train_ds = MonteCarloDenoiseDataset(config.data_dir, "train", transform=train_tf, load_auxiliary=config.load_auxiliary)
        val_ds = MonteCarloDenoiseDataset(config.data_dir, "val", transform=val_tf, load_auxiliary=config.load_auxiliary)
        
        if use_crop:
            print(f"Using crop_size: {config.crop_size}x{config.crop_size} (with RandomCrop)")
        else:
            print(f"Resizing to: {config.crop_size}x{config.crop_size} (no RandomCrop, only ResizeTo)")

    train_loader = create_data_loader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, backend=backend)
    val_loader = create_data_loader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, backend=backend)

    UNetClass = get_unet(backend)
    model = UNetClass(
        config.n_channels, 
        config.n_classes, 
        config.bilinear,
        use_attention=getattr(config, 'use_attention', True),
        use_residual_learning=getattr(config, 'use_residual_learning', True)
    )
    if backend == 'jittor':
        criterion = get_loss_function_jt(config.loss_type, weights=getattr(config, "loss_weights", None))
        optimizer = jt_optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        criterion = get_loss_function_torch(config.loss_type, weights=getattr(config, "loss_weights", None))
        optimizer = torch_optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        model.to(device)

    best_psnr = -1
    best_metric = -float("inf")
    no_improve = 0
    start_epoch = 0
    if config.resume_from and os.path.exists(config.resume_from):
        if backend == 'jittor':
            ckpt = jt.load(config.resume_from)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"]
            best_psnr = ckpt.get("metrics", {}).get("psnr", best_psnr)
            best_metric = ckpt.get("metrics", {}).get(config.early_stop_metric, best_metric)
        else:
            import torch
            ckpt = torch.load(config.resume_from, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"]
            best_psnr = ckpt.get("metrics", {}).get("psnr", best_psnr)
            best_metric = ckpt.get("metrics", {}).get(config.early_stop_metric, best_metric)
        print(f"Resume from epoch {start_epoch}")

    train_batches = math.ceil(len(train_ds) / config.batch_size)

    base_lr = config.learning_rate
    eta_min = config.scheduler_params.get("cosine", {}).get("eta_min", 1e-6)
    T_max = config.scheduler_params.get("cosine", {}).get("T_max", config.num_epochs)
    step_size = config.scheduler_params.get("step", {}).get("step_size", 30)
    gamma = config.scheduler_params.get("step", {}).get("gamma", 0.1)

    for epoch in range(start_epoch, config.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{config.num_epochs} ===")
        if backend == 'jittor':
            train_loss = train_epoch_jittor(model, train_loader, criterion, optimizer, epoch+1, config, total_batches=train_batches)
            val_loss, val_metrics = validate_jittor(model, val_loader, criterion)
        else:
            train_loss = train_epoch_torch(model, train_loader, criterion, optimizer, epoch+1, config, total_batches=train_batches, device=device)
            val_loss, val_metrics = validate_torch(model, val_loader, criterion, device=device)
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val   Loss: {val_loss:.6f}")
        print_metrics(val_metrics, prefix="Val")

        current_metric = val_metrics.get(config.early_stop_metric, -val_loss if config.early_stop_metric == "val_loss" else val_metrics.get("psnr", -val_loss))
        if current_metric > best_metric:
            best_metric = current_metric
            no_improve = 0
        else:
            no_improve += 1

        is_best = val_metrics.get("psnr", -float("inf")) > best_psnr
        if is_best:
            best_psnr = val_metrics["psnr"]
        if ((epoch + 1) % config.save_freq == 0) or is_best:
            os.makedirs(config.save_dir, exist_ok=True)
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": val_metrics,
            }
            latest = os.path.join(config.save_dir, "latest_model.pkl")
            if backend == 'jittor':
                jt.save(ckpt, latest)
                if is_best:
                    jt.save(ckpt, os.path.join(config.save_dir, "best_model.pkl"))
            else:
                import torch
                torch.save(ckpt, latest)
                if is_best:
                    torch.save(ckpt, os.path.join(config.save_dir, "best_model.pkl"))
            print(f"Checkpoint saved at epoch {epoch+1}")

        # LR schedule
        if config.scheduler == "cosine":
            lr_new = eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * (epoch + 1) / T_max))
        elif config.scheduler == "step":
            lr_new = base_lr * (gamma ** ((epoch + 1) // step_size))
        else:
            lr_new = base_lr
        optimizer.lr = lr_new
        print(f"[LR] current lr: {optimizer.lr:.6e}")

        if no_improve >= config.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {no_improve} epochs)")
            break

    print("Training done.")


if __name__ == "__main__":
    main()


import argparse
import os
from typing import Optional

import jittor as jt
import jittor.optim as optim

from data.dataset import MCDenoiseDataset
from losses.denoise_loss import denoise_l1_loss, calc_psnr
from models.resnet_denoiser import ResidualDenoiser
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.common import ensure_dir, load_config, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Jittor Monte Carlo 图像降噪 (CPU)")
    parser.add_argument("--config", type=str, default="configs/default.py", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="检查点路径")
    parser.add_argument("--clean_root", type=str, default=None, help="训练 clean 目录，覆盖配置")
    parser.add_argument("--noisy_root", type=str, default=None, help="训练 noisy 目录，覆盖配置")
    parser.add_argument("--val_clean_root", type=str, default=None, help="验证 clean 目录，覆盖配置")
    parser.add_argument("--val_noisy_root", type=str, default=None, help="验证 noisy 目录，覆盖配置")
    parser.add_argument("--num_epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖 batch size")
    parser.add_argument("--noise_sigma", type=float, default=None, help="覆盖噪声增强强度")
    return parser.parse_args()


def make_dataloader(cfg: dict, clean_root: str, noisy_root: str, shuffle: bool) -> Optional[MCDenoiseDataset]:
    if not clean_root or not noisy_root:
        return None
    if not os.path.isdir(clean_root):
        raise FileNotFoundError(f"clean 目录不存在: {clean_root}")
    if not os.path.isdir(noisy_root):
        raise FileNotFoundError(f"noisy 目录不存在: {noisy_root}")
    dataset = MCDenoiseDataset(
        clean_root=clean_root,
        noisy_root=noisy_root,
        image_size=cfg["image_size"],
        num_mc_samples=cfg["num_mc_samples"],
        max_mc_samples=cfg["max_mc_samples"],
        noise_sigma=cfg["noise_sigma"],
    )
    return dataset.set_attrs(
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["num_workers"],
    )


def validate(model, val_loader) -> tuple:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    count = 0
    with jt.no_grad():
        for noisy, clean in val_loader:
            pred = model(noisy)
            loss = denoise_l1_loss(pred, clean)
            total_loss += float(loss.item())
            total_psnr += calc_psnr(pred, clean)
            count += 1
    model.train()
    if count == 0:
        return 0.0, 0.0
    return total_loss / count, total_psnr / count


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 覆盖部分配置
    if args.clean_root:
        cfg["clean_root"] = args.clean_root
    if args.noisy_root:
        cfg["noisy_root"] = args.noisy_root
    if args.val_clean_root:
        cfg["val_clean_root"] = args.val_clean_root
    if args.val_noisy_root:
        cfg["val_noisy_root"] = args.val_noisy_root
    if args.num_epochs:
        cfg["num_epochs"] = args.num_epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.noise_sigma is not None:
        cfg["noise_sigma"] = args.noise_sigma

    # 强制 CPU
    jt.flags.use_cuda = 0
    set_random_seed(cfg["seed"])
    ensure_dir(cfg["save_dir"])

    print(f"使用配置: {cfg}")

    model = ResidualDenoiser(
        in_channels=4,
        base_channels=cfg["base_channels"],
        out_channels=3,
        num_blocks=5,
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    start_epoch = 0
    if args.resume:
        state = load_checkpoint(model, optimizer, args.resume)
        start_epoch = int(state.get("epoch", 0)) + 1
        print(f"从检查点 {args.resume} 恢复，起始 epoch={start_epoch}")

    train_loader = make_dataloader(cfg, cfg["clean_root"], cfg["noisy_root"], shuffle=True)
    val_loader = None
    if cfg.get("val_clean_root") and cfg.get("val_noisy_root"):
        val_loader = make_dataloader(cfg, cfg["val_clean_root"], cfg["val_noisy_root"], shuffle=False)
    if train_loader is None:
        raise ValueError("未提供训练数据目录 clean_root/noisy_root")

    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, cfg["num_epochs"]):
        for batch_idx, (noisy, clean) in enumerate(train_loader):
            pred = model(noisy)
            loss = denoise_l1_loss(pred, clean)
            optimizer.step(loss)
            global_step += 1

            if (batch_idx + 1) % cfg["log_interval"] == 0:
                print(
                    f"[Epoch {epoch+1}/{cfg['num_epochs']}] "
                    f"Step {global_step} | Loss: {float(loss.item()):.4f}"
                )

        if val_loader and (epoch + 1) % cfg["val_interval"] == 0:
            val_loss, val_psnr = validate(model, val_loader)
            print(f"验证 | Epoch {epoch+1}: loss={val_loss:.4f}, psnr={val_psnr:.2f} dB")

        ckpt_path = os.path.join(cfg["save_dir"], f"epoch_{epoch+1}.pkl")
        save_checkpoint(model, optimizer, epoch, ckpt_path)
        print(f"已保存检查点: {ckpt_path}")


if __name__ == "__main__":
    main()


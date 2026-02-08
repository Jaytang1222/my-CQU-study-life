import argparse
import os

import jittor as jt
import numpy as np
from PIL import Image

from data.dataset import MCDenoiseDataset
from losses.denoise_loss import denoise_l1_loss, calc_psnr
from models.resnet_denoiser import ResidualDenoiser
from utils.common import ensure_dir, load_config, set_random_seed
from utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="验证与可视化 (CPU)")
    parser.add_argument("--config", type=str, default="configs/default.py", help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径")
    parser.add_argument("--clean_root", type=str, default=None, help="验证 clean 目录，覆盖配置")
    parser.add_argument("--noisy_root", type=str, default=None, help="验证 noisy 目录，覆盖配置")
    parser.add_argument("--save_dir", type=str, default="outputs/val_vis", help="可视化输出目录")
    parser.add_argument("--num_save", type=int, default=5, help="保存前多少张可视化")
    return parser.parse_args()


def make_val_loader(cfg, clean_root, noisy_root):
    dataset = MCDenoiseDataset(
        clean_root=clean_root,
        noisy_root=noisy_root,
        image_size=cfg["image_size"],
        num_mc_samples=cfg["num_mc_samples"],
        max_mc_samples=cfg["max_mc_samples"],
        noise_sigma=cfg["noise_sigma"],
    )
    return dataset.set_attrs(
        batch_size=1,
        shuffle=False,
        num_workers=cfg["num_workers"],
    )


def tensor_to_pil(t: jt.Var) -> Image.Image:
    """
    t: C,H,W in [0,1]
    """
    np_img = np.clip(np.transpose(t.numpy(), (1, 2, 0)), 0, 1)
    np_img = (np_img * 255.0).astype(np.uint8)
    return Image.fromarray(np_img)


def save_triplet(noisy, pred, clean, path):
    n_img = tensor_to_pil(noisy)
    p_img = tensor_to_pil(pred)
    c_img = tensor_to_pil(clean)
    # 横向拼接
    w, h = n_img.size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(n_img, (0, 0))
    canvas.paste(p_img, (w, 0))
    canvas.paste(c_img, (w * 2, 0))
    canvas.save(path)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.clean_root:
        cfg["val_clean_root"] = args.clean_root
    if args.noisy_root:
        cfg["val_noisy_root"] = args.noisy_root
    if not cfg.get("val_clean_root") or not cfg.get("val_noisy_root"):
        raise ValueError("必须提供验证 clean/noisy 目录")

    jt.flags.use_cuda = 0
    set_random_seed(cfg["seed"])
    ensure_dir(args.save_dir)

    # 模型
    model = ResidualDenoiser(
        in_channels=4,
        base_channels=cfg["base_channels"],
        out_channels=3,
        num_blocks=5,
    )
    optimizer = None  # 占位，load_checkpoint 需要
    load_checkpoint(model, optimizer, args.checkpoint)
    model.eval()

    val_loader = make_val_loader(cfg, cfg["val_clean_root"], cfg["val_noisy_root"])

    total_l1 = 0.0
    total_psnr = 0.0
    count = 0

    with jt.no_grad():
        for idx, (model_input, clean) in enumerate(val_loader):
            noisy = model_input[:, :3, :, :] if model_input.ndim == 4 else model_input[:3]
            pred = model(model_input)
            loss = denoise_l1_loss(pred, clean)

            total_l1 += float(loss.item())
            total_psnr += calc_psnr(pred, clean)
            count += 1

            if idx < args.num_save:
                out_path = os.path.join(args.save_dir, f"vis_{idx+1}.png")
                save_triplet(noisy[0], pred[0], clean[0], out_path)

    if count == 0:
        print("验证集中没有数据")
        return

    avg_l1 = total_l1 / count
    avg_psnr = total_psnr / count
    print(f"验证集: L1={avg_l1:.4f}, PSNR={avg_psnr:.2f} dB, 样本数={count}")
    print(f"前 {min(args.num_save, count)} 张可视化已保存在: {args.save_dir}")


if __name__ == "__main__":
    main()


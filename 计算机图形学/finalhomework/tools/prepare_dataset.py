"""
整理数据集到标准结构：
dataset/
  train/val/test/
    noisy_images/
    clean_images/

支持两种输入：
1) 直接 noisy/clean 子目录 (--nested False)
2) 嵌套子目录，每个子目录内一对含噪/干净图 (--nested True)
   通过文件名关键字匹配，如 NOISY_SRGB / GT_SRGB

支持将大图裁剪成多个小图（--crop_to_patches），增加样本数量
"""
import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import math


def list_pairs_flat(noisy_dir: Path, clean_dir: Path) -> List[Tuple[Path, Path, str]]:
    noisy_files = {p.name: p for p in noisy_dir.iterdir() if p.is_file()}
    clean_files = {p.name: p for p in clean_dir.iterdir() if p.is_file()}
    common = sorted(set(noisy_files) & set(clean_files))
    if not common:
        raise RuntimeError("未找到同名的 noisy/clean 配对文件")
    return [(noisy_files[n], clean_files[n], n) for n in common]


def list_pairs_nested(root: Path, noisy_key: str, clean_key: str) -> List[Tuple[Path, Path, str]]:
    pairs = []
    # 跳过隐藏目录和Python缓存目录
    skip_dirs = {'__pycache__', '.git', '.ipynb_checkpoints', '.DS_Store'}
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        # 跳过隐藏目录和缓存目录
        if sub.name.startswith('.') or sub.name.startswith('__') or sub.name in skip_dirs:
            continue
        files = [f for f in sub.iterdir() if f.is_file()]
        noisy = [f for f in files if noisy_key.lower() in f.name.lower()]
        clean = [f for f in files if clean_key.lower() in f.name.lower()]
        if not noisy or not clean:
            print(f"[跳过] 子目录缺少 noisy/clean: {sub}")
            continue
        out_name = f"{sub.name}{noisy[0].suffix.lower()}"
        pairs.append((noisy[0], clean[0], out_name))
    if not pairs:
        raise RuntimeError("未在嵌套目录中找到成对文件，请检查关键字")
    return pairs


def split_pairs(pairs, train_ratio, val_ratio):
    n = len(pairs)
    t_end = int(n * train_ratio)
    v_end = t_end + int(n * val_ratio)
    return pairs[:t_end], pairs[t_end:v_end], pairs[v_end:]


def ensure_dirs(out_root: Path):
    for split in ["train", "val", "test"]:
        (out_root / split / "noisy_images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "clean_images").mkdir(parents=True, exist_ok=True)


def crop_image_to_patches(img_path: Path, num_patches: int = 16) -> List[Image.Image]:
    """
    将图片裁剪成多个patch
    Args:
        img_path: 图片路径
        num_patches: patch数量（默认16，即4x4网格）
    Returns:
        patch列表
    """
    # 使用懒加载，避免一次性加载整个大图到内存
    with Image.open(img_path) as img_file:
        img = img_file.convert('RGB')
        w, h = img.size
    
    # 计算网格大小（尽量接近正方形）
    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        # 如果不是完全平方数，使用最接近的配置
        grid_size = int(math.sqrt(num_patches))
        # 例如16=4x4, 9=3x3, 12=3x4等
        cols = grid_size
        rows = (num_patches + cols - 1) // cols
    else:
        rows = cols = grid_size
    
    # 计算每个patch的尺寸
    patch_w = w // cols
    patch_h = h // rows
    
    patches = []
    for i in range(rows):
        for j in range(cols):
            if len(patches) >= num_patches:
                break
            # 计算裁剪区域
            left = j * patch_w
            top = i * patch_h
            right = left + patch_w
            bottom = top + patch_h
            
            # 裁剪（复制数据，避免引用原图）
            patch = img.crop((left, top, right, bottom)).copy()
            patches.append(patch)
        if len(patches) >= num_patches:
            break
    
    return patches


def transfer(src: Path, dst: Path, symlink: bool, crop_patches: int = None):
    """
    复制或链接文件，可选裁剪成patches
    Args:
        src: 源文件路径
        dst: 目标文件路径
        symlink: 是否使用软链接
        crop_patches: 如果指定，将图片裁剪成多个patch（数量）
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    if crop_patches and crop_patches > 1:
        # 裁剪成多个patch
        try:
            patches = crop_image_to_patches(src, crop_patches)
            base_name = dst.stem
            suffix = dst.suffix
            
            for idx, patch in enumerate(patches):
                patch_dst = dst.parent / f"{base_name}_patch{idx:02d}{suffix}"
                # 使用optimize=False加快保存速度，quality=95保持质量
                patch.save(patch_dst, quality=95, optimize=False)
        except Exception as e:
            print(f"[警告] 裁剪失败 {src}: {e}，使用原图")
            if symlink:
                if dst.exists():
                    dst.unlink()
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)
    else:
        # 正常复制或链接
        if symlink:
            if dst.exists():
                dst.unlink()
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原始数据根目录")
    ap.add_argument("--out", default="./dataset", help="输出目录")
    ap.add_argument("--nested", action="store_true", default=True, help="是否为嵌套子目录结构")
    ap.add_argument("--noisy-subdir", default="noisy", help="flat 模式下含噪子目录")
    ap.add_argument("--clean-subdir", default="clean", help="flat 模式下干净子目录")
    ap.add_argument("--noisy-key", default="NOISY_SRGB", help="nested 模式 noisy 文件名关键字")
    ap.add_argument("--clean-key", default="GT_SRGB", help="nested 模式 clean 文件名关键字")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--symlink", action="store_true", help="使用软链接而非复制")
    ap.add_argument("--crop-to-patches", type=int, default=None, 
                    help="将每张图裁剪成N个小图（例如16表示4x4网格），增加样本数量")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    if args.nested:
        pairs = list_pairs_nested(src, args.noisy_key, args.clean_key)
    else:
        noisy_dir = src / args.noisy_subdir
        clean_dir = src / args.clean_subdir
        if not noisy_dir.is_dir() or not clean_dir.is_dir():
            raise RuntimeError(f"未找到 noisy/clean 目录: {noisy_dir} / {clean_dir}")
        pairs = list_pairs_flat(noisy_dir, clean_dir)

    train_pairs, val_pairs, test_pairs = split_pairs(pairs, args.train_ratio, args.val_ratio)
    ensure_dirs(out)

    def dump(split, items):
        count = 0
        total = len(items)
        print(f"\n处理 {split} 集: {total} 张图片...")
        for idx, (noisy, clean, name) in enumerate(items, 1):
            print(f"  [{idx}/{total}] 处理 {name}...", end="", flush=True)
            try:
                if args.crop_to_patches:
                    # 裁剪模式：每张图生成多个patch
                    transfer(noisy, out / split / "noisy_images" / name, args.symlink, args.crop_to_patches)
                    transfer(clean, out / split / "clean_images" / name, args.symlink, args.crop_to_patches)
                    count += args.crop_to_patches
                    print(f" ✓ (生成 {args.crop_to_patches} 个patch)")
                else:
                    # 正常模式
                    transfer(noisy, out / split / "noisy_images" / name, args.symlink)
                    transfer(clean, out / split / "clean_images" / name, args.symlink)
                    count += 1
                    print(f" ✓")
            except Exception as e:
                print(f" ✗ 错误: {e}")
                continue
        return count

    print("=" * 60)
    print(f"找到 {len(pairs)} 对图片")
    print(f"划分: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    if args.crop_to_patches:
        print(f"每张图将裁剪成 {args.crop_to_patches} 个小图")
    print("=" * 60)
    
    train_count = dump("train", train_pairs)
    val_count = dump("val", val_pairs)
    test_count = dump("test", test_pairs)

    print("=" * 60)
    print("整理完成")
    print(f"原始图片数: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    if args.crop_to_patches:
        print(f"裁剪后图片数: train={train_count} val={val_count} test={test_count}")
        print(f"每张原图裁剪成 {args.crop_to_patches} 个小图")
        print(f"样本数量增加: {args.crop_to_patches}x")
    else:
        print(f"最终图片数: train={train_count} val={val_count} test={test_count}")
    if args.symlink:
        print("已使用软链接")
    print("=" * 60)


if __name__ == "__main__":
    main()


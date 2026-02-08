import os
import shutil
import random

# --- 配置参数 ---

# 包含所有原始图像的根目录（例如：CroppedImages）
SOURCE_ROOT = r'C:\Users\jayta\Desktop\study\qhx\finalhomework\CroppedImages'

# 目标数据集的根目录，脚本将在这里创建 train 和 val 文件夹
DESTINATION_ROOT = r'C:\Users\jayta\Desktop\study\qhx\finalhomework\dataset'

# 训练集和验证集的划分比例 (例如: 0.8 代表 80% 训练集, 20% 验证集)
TRAIN_SPLIT_RATIO = 0.8


# --- 主要函数 ---

def organize_dataset(source_dir, dest_dir, split_ratio):
    """
    遍历源目录中的图片，根据文件名后缀进行分类（clean/noisy），
    并按比例划分到目标目录的 train/val 子目录中。
    """
    print(f"🚀 开始处理目录: {source_dir}")

    # 1. 定义目标文件夹路径
    train_clean_dir = os.path.join(dest_dir, 'train', 'clean')
    train_noisy_dir = os.path.join(dest_dir, 'train', 'noisy')
    val_clean_dir = os.path.join(dest_dir, 'val', 'clean')
    val_noisy_dir = os.path.join(dest_dir, 'val', 'noisy')

    # 确保所有目标文件夹存在
    os.makedirs(train_clean_dir, exist_ok=True)
    os.makedirs(train_noisy_dir, exist_ok=True)
    os.makedirs(val_clean_dir, exist_ok=True)
    os.makedirs(val_noisy_dir, exist_ok=True)

    # 2. 收集文件对
    # 使用字典来存储 (文件名公共部分: 文件路径) 的映射
    # 例如: 'Canon5D2_5_160_3200_chair_5': ['..._mean.JPG', '..._real.JPG']
    file_pairs = {}

    # 遍历源目录中的所有文件
    for filename in os.listdir(source_dir):
        if filename.endswith(('_mean.JPG', '_real.JPG')):
            full_path = os.path.join(source_dir, filename)

            # 提取文件名公共部分（不包含 _mean.JPG 或 _real.JPG）
            # 例如 'Canon5D2_5_160_3200_chair_5_mean.JPG' -> 'Canon5D2_5_160_3200_chair_5'
            base_name = filename.rsplit('_', 1)[0]

            if base_name not in file_pairs:
                file_pairs[base_name] = []

            file_pairs[base_name].append(full_path)

    # 3. 筛选出有效的 clean/noisy 图像对
    valid_pairs = []
    for base_name, paths in file_pairs.items():
        # 必须同时存在 mean (clean) 和 real (noisy) 文件才能构成有效对
        mean_path = next((p for p in paths if p.endswith('_mean.JPG')), None)
        real_path = next((p for p in paths if p.endswith('_real.JPG')), None)

        if mean_path and real_path:
            valid_pairs.append({'base_name': base_name, 'clean': mean_path, 'noisy': real_path})

    print(f"✅ 找到 {len(valid_pairs)} 对有效的 clean/noisy 图像。")
    if not valid_pairs:
        print("❌ 未找到任何图像对。请检查 SOURCE_ROOT 和文件命名格式是否正确。")
        return

    # 4. 随机划分数据集
    random.shuffle(valid_pairs)
    split_index = int(len(valid_pairs) * split_ratio)

    train_pairs = valid_pairs[:split_index]
    val_pairs = valid_pairs[split_index:]

    print(f"📊 划分结果: 训练集 {len(train_pairs)} 对, 验证集 {len(val_pairs)} 对。")

    # 5. 复制文件到目标目录
    def copy_files(pairs, clean_dest, noisy_dest):
        """将图像对复制到指定的 clean 和 noisy 目标目录"""
        count = 0
        for pair in pairs:
            base_name = pair['base_name']
            clean_src = pair['clean']
            noisy_src = pair['noisy']

            # 使用原始文件名作为目标文件名
            clean_dest_path = os.path.join(clean_dest, os.path.basename(clean_src))
            noisy_dest_path = os.path.join(noisy_dest, os.path.basename(noisy_src))

            try:
                # 复制 clean 图像
                shutil.copy2(clean_src, clean_dest_path)
                # 复制 noisy 图像
                shutil.copy2(noisy_src, noisy_dest_path)
                count += 1
            except Exception as e:
                print(f"⚠️ 复制文件时出错 {base_name}: {e}")
        return count

    print("\n📦 正在复制训练集文件...")
    train_count = copy_files(train_pairs, train_clean_dir, train_noisy_dir)

    print("📦 正在复制验证集文件...")
    val_count = copy_files(val_pairs, val_clean_dir, val_noisy_dir)

    print("\n🎉 数据集整理完成!")
    print(f"   - 训练集图像数: {train_count * 2} (clean: {train_count}, noisy: {train_count})")
    print(f"   - 验证集图像数: {val_count * 2} (clean: {val_count}, noisy: {val_count})")
    print(f"   - 目标目录结构如下：")

    # 打印最终目录结构
    print(f"     {DESTINATION_ROOT}")
    print(f"     ├── train")
    print(f"     │   ├── clean (共 {train_count} 张 GT 图)")
    print(f"     │   └── noisy (共 {train_count} 张带噪图)")
    print(f"     └── val")
    print(f"         ├── clean (共 {val_count} 张 GT 图)")
    print(f"         └── noisy (共 {val_count} 张带噪图)")


if __name__ == '__main__':
    organize_dataset(SOURCE_ROOT, DESTINATION_ROOT, TRAIN_SPLIT_RATIO)
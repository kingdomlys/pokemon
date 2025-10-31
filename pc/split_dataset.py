"""
宝可梦数据集划分脚本
将Dataset_pokemon按照 train:val:test = 70:15:15 的比例划分
适用于YOLOv8-cls训练
"""
import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    划分数据集
    
    Args:
        source_dir: 原始数据集路径 (Dataset_pokemon)
        target_dir: 目标数据集路径 (Dataset_pokemon_split)
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 创建目标目录结构
    splits = ['train', 'val', 'test']
    for split in splits:
        (target_path / split).mkdir(parents=True, exist_ok=True)
    
    # 获取所有类别文件夹
    class_folders = [f for f in source_path.iterdir() if f.is_dir()]
    class_folders.sort()  # 排序保证一致性
    
    print(f"发现 {len(class_folders)} 个宝可梦类别")
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for class_folder in class_folders:
        class_name = class_folder.name
        print(f"\n处理类别: {class_name}")
        
        # 获取该类别下所有图片
        images = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
        
        if len(images) == 0:
            print(f"  警告: {class_name} 没有图片,跳过")
            continue
        
        # 随机打乱
        random.shuffle(images)
        
        # 计算划分点
        n_images = len(images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        n_test = n_images - n_train - n_val  # 剩余的都给测试集
        
        # 划分图片
        train_images = images[:n_train]
        val_images = images[n_train:n_train+n_val]
        test_images = images[n_train+n_val:]
        
        # 复制图片到对应目录
        for split_name, split_images in [('train', train_images), 
                                          ('val', val_images), 
                                          ('test', test_images)]:
            split_class_dir = target_path / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in split_images:
                shutil.copy2(img_path, split_class_dir / img_path.name)
        
        total_train += len(train_images)
        total_val += len(val_images)
        total_test += len(test_images)
        
        print(f"  总计: {n_images} | 训练: {len(train_images)} | 验证: {len(val_images)} | 测试: {len(test_images)}")
    
    print(f"\n" + "="*60)
    print(f"数据集划分完成!")
    print(f"训练集: {total_train} 张图片")
    print(f"验证集: {total_val} 张图片")
    print(f"测试集: {total_test} 张图片")
    print(f"总计: {total_train + total_val + total_test} 张图片")
    print(f"输出目录: {target_path.absolute()}")
    print("="*60)

if __name__ == "__main__":
    # 设置随机种子以保证可复现
    random.seed(42)
    
    # 配置路径
    SOURCE_DIR = "Dataset_pokemon"
    TARGET_DIR = "Dataset_pokemon_split"
    
    # 执行划分
    split_dataset(
        source_dir=SOURCE_DIR,
        target_dir=TARGET_DIR,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print("\n提示: 可以使用以下命令训练模型:")
    print(f"python train_cls.py")

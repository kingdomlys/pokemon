"""
YOLOv8-cls 宝可梦分类模型测试脚本
测试模型在测试集上的性能
"""
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

def test_on_testset(model_path, test_data_dir):
    """
    在测试集上评估模型
    
    Args:
        model_path: 模型路径
        test_data_dir: 测试集路径
    """
    print("="*60)
    print("🧪 宝可梦分类模型测试")
    print("="*60)
    
    # 加载模型
    print(f"\n📦 加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 验证模型
    print(f"\n📊 在测试集上评估...")
    metrics = model.val(data=test_data_dir, split='test')
    
    print(f"\n测试集性能:")
    print(f"  Top-1 准确率: {metrics.top1:.4f}")
    print(f"  Top-5 准确率: {metrics.top5:.4f}")
    
    return metrics

def test_single_image(model_path, image_path, show=True):
    """
    测试单张图片
    
    Args:
        model_path: 模型路径
        image_path: 图片路径
        show: 是否显示结果
    """
    print(f"\n🖼️  测试图片: {image_path}")
    
    # 加载模型
    model = YOLO(model_path)
    
    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return None
    
    # 预测
    results = model(img)[0]
    names = results.names
    
    # 获取预测结果
    top1_label = results.probs.top1
    top5_labels = results.probs.top5
    top1_conf = results.probs.top1conf.item()
    top5_conf = results.probs.top5conf.cpu().numpy()
    
    top1_name = names[top1_label]
    
    print(f"\n✅ 预测结果:")
    print(f"   Top-1: {top1_name} (置信度: {top1_conf:.4f})")
    print(f"\n   Top-5:")
    for i, (label, conf) in enumerate(zip(top5_labels, top5_conf), 1):
        print(f"   {i}. {names[label]:20s} - {conf:.4f}")
    
    # 可视化
    if show:
        visualize_prediction(img, top1_name, top1_conf, names, top5_labels, top5_conf)
    
    return {
        'top1_name': top1_name,
        'top1_conf': top1_conf,
        'top5_names': [names[label] for label in top5_labels],
        'top5_conf': top5_conf
    }

def visualize_prediction(img, top1_name, top1_conf, names, top5_labels, top5_conf):
    """
    可视化预测结果
    """
    # 转换BGR到RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 显示图片
    ax1.imshow(img_rgb)
    ax1.axis('off')
    ax1.set_title(f'预测: {top1_name}\n置信度: {top1_conf:.4f}', fontsize=14, fontweight='bold')
    
    # 显示Top-5概率
    top5_names = [names[label] for label in top5_labels]
    y_pos = np.arange(len(top5_names))
    
    ax2.barh(y_pos, top5_conf, color='skyblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top5_names)
    ax2.invert_yaxis()
    ax2.set_xlabel('置信度', fontsize=12)
    ax2.set_title('Top-5 预测', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    
    # 添加数值标签
    for i, v in enumerate(top5_conf):
        ax2.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.show()

def batch_test_samples(model_path, test_data_dir, n_samples=5):
    """
    批量测试样本
    
    Args:
        model_path: 模型路径
        test_data_dir: 测试集路径
        n_samples: 每个类别测试的样本数
    """
    print(f"\n🎲 随机测试 {n_samples} 个样本...")
    
    test_path = Path(test_data_dir) / 'test'
    
    if not test_path.exists():
        print(f"❌ 测试集不存在: {test_path}")
        return
    
    # 获取所有类别
    class_folders = [f for f in test_path.iterdir() if f.is_dir()]
    
    if len(class_folders) == 0:
        print("❌ 没有找到测试类别")
        return
    
    # 随机选择几个类别
    import random
    random.seed(42)
    selected_classes = random.sample(class_folders, min(n_samples, len(class_folders)))
    
    correct = 0
    total = 0
    
    for class_folder in selected_classes:
        class_name = class_folder.name
        images = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
        
        if len(images) == 0:
            continue
        
        # 随机选择一张图片
        img_path = random.choice(images)
        
        print(f"\n" + "-"*60)
        print(f"真实类别: {class_name}")
        
        result = test_single_image(model_path, img_path, show=False)
        
        if result and class_name in result['top1_name']:
            correct += 1
            print("✅ 预测正确!")
        else:
            print("❌ 预测错误!")
        
        total += 1
    
    print(f"\n" + "="*60)
    print(f"抽样准确率: {correct}/{total} = {correct/total*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    # 配置
    MODEL_PATH = "runs/classify/pokemon_yolov8n/weights/best.pt"  # 训练好的模型
    TEST_DATA_DIR = "Dataset_pokemon_split"  # 数据集根目录
    
    # 1. 在完整测试集上评估
    print("\n【任务1】完整测试集评估")
    test_on_testset(MODEL_PATH, TEST_DATA_DIR)
    
    # 2. 批量抽样测试
    print("\n\n【任务2】随机抽样测试")
    batch_test_samples(MODEL_PATH, TEST_DATA_DIR, n_samples=10)
    
    # 3. 测试单张图片(示例)
    print("\n\n【任务3】单张图片测试")
    sample_img = "Dataset_pokemon/0001/0001Bulbasaur1.jpg"
    if Path(sample_img).exists():
        test_single_image(MODEL_PATH, sample_img, show=True)

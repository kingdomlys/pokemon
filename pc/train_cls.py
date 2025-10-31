"""
YOLOv8-cls 宝可梦分类模型训练脚本
适用于树莓派4B部署
"""
from ultralytics import YOLO
import torch

def train_pokemon_classifier():
    """
    训练宝可梦分类模型
    """
    print("="*60)
    print("🎮 宝可梦图鉴 AI 训练程序")
    print("="*60)
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📱 使用设备: {device}")
    
    # 加载预训练模型
    print("\n📦 加载YOLOv8n-cls预训练模型...")
    model = YOLO('pretrain/yolov8n-cls.pt')
    
    # 训练参数配置
    print("\n⚙️  配置训练参数...")
    training_args = {
        'data': 'Dataset_pokemon_split',  # 数据集路径
        'epochs': 100,                     # 训练轮数
        'batch': 32,                       # 批次大小(根据内存调整)
        'imgsz': 224,                      # 图像大小
        'device': device,                  # 设备
        'workers': 4,                      # 数据加载线程数
        'optimizer': 'Adam',               # 优化器
        'lr0': 0.001,                      # 初始学习率
        'patience': 20,                    # 早停耐心值
        'save': True,                      # 保存模型
        'save_period': 10,                 # 每10轮保存一次
        'project': 'runs/classify',        # 项目目录
        'name': 'pokemon_yolov8n',         # 实验名称
        'exist_ok': True,                  # 允许覆盖
        'pretrained': True,                # 使用预训练权重
        'verbose': True,                   # 详细输出
    }
    
    # 显示配置
    print("\n训练配置:")
    for key, value in training_args.items():
        print(f"  {key:15s}: {value}")
    
    # 开始训练
    print("\n🚀 开始训练...\n")
    results = model.train(**training_args)
    
    # 训练完成
    print("\n" + "="*60)
    print("✅ 训练完成!")
    print("="*60)
    
    # 验证模型
    print("\n📊 在验证集上评估模型...")
    metrics = model.val()
    
    print(f"\n模型性能:")
    print(f"  Top-1 准确率: {metrics.top1:.4f}")
    print(f"  Top-5 准确率: {metrics.top5:.4f}")
    
    # 导出模型(用于树莓派部署)
    print("\n📤 导出模型用于部署...")
    
    # 导出为ONNX格式(推荐用于树莓派)
    print("  导出ONNX格式...")
    onnx_path = model.export(format='onnx', imgsz=224, simplify=True)
    print(f"  ✅ ONNX模型: {onnx_path}")
    
    # 也可以导出TorchScript格式
    # print("  导出TorchScript格式...")
    # ts_path = model.export(format='torchscript', imgsz=224)
    # print(f"  ✅ TorchScript模型: {ts_path}")
    
    print("\n" + "="*60)
    print("🎉 所有任务完成!")
    print("="*60)
    print(f"\n最佳模型保存在: runs/classify/pokemon_yolov8n/weights/best.pt")
    print(f"ONNX模型用于树莓派部署: {onnx_path}")
    print("\n下一步: 使用 python test_cls.py 测试模型")

if __name__ == "__main__":
    train_pokemon_classifier()

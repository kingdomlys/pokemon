"""
树莓派交互式宝可梦识别脚本
支持多种验证方式:
1. 单张图片识别
2. 批量图片识别
3. 目录遍历识别
4. 交互式命令行
"""
import cv2
import numpy as np
from pathlib import Path
import time
import argparse
from deploy_raspberry_pi import PokemonPokedex

class InteractivePokedex:
    """交互式宝可梦图鉴"""
    
    def __init__(self, model_path, names_file, conf_threshold=0.5):
        """初始化交互式图鉴"""
        print("="*60)
        print("🎮 宝可梦图鉴 - 交互式识别系统")
        print("="*60)
        
        # 初始化识别器
        self.pokedex = PokemonPokedex(model_path, names_file, conf_threshold)
        self.history = []  # 识别历史
    
    def predict_single(self, image_path, show_image=False):
        """
        识别单张图片
        
        Args:
            image_path: 图片路径
            show_image: 是否显示图片(需要图形界面)
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"❌ 文件不存在: {image_path}")
            return None
        
        print(f"\n{'='*60}")
        print(f"📸 正在识别: {image_path.name}")
        print(f"{'='*60}")
        
        # 预测
        result = self.pokedex.predict(str(image_path), verbose=True)
        
        # 保存历史
        self.history.append({
            'file': str(image_path),
            'result': result
        })
        
        # 显示图片(如果支持)
        if show_image:
            try:
                img = cv2.imread(str(image_path))
                if img is not None:
                    # 添加预测结果到图片
                    img_display = self._add_text_to_image(img, result)
                    cv2.imshow('Pokemon Detection', img_display)
                    print("\n💡 按任意键继续...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            except Exception as e:
                print(f"⚠️  无法显示图片(可能是无图形界面): {e}")
        
        return result
    
    def predict_batch(self, image_paths, max_display=10):
        """
        批量识别
        
        Args:
            image_paths: 图片路径列表
            max_display: 最多显示的结果数
        """
        print(f"\n{'='*60}")
        print(f"📦 批量识别模式 - 共 {len(image_paths)} 张图片")
        print(f"{'='*60}\n")
        
        results = []
        start_time = time.time()
        
        for i, img_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] ", end="")
            
            try:
                result = self.pokedex.predict(str(img_path), verbose=False)
                results.append({
                    'file': Path(img_path).name,
                    'path': str(img_path),
                    'result': result
                })
                
                # 简要输出
                pokemon_name = result.get('top1_name', f"ID:{result['top1_label']}")
                conf = result['top1_conf']
                print(f"{Path(img_path).name:40s} -> {pokemon_name:20s} ({conf:.4f})")
                
            except Exception as e:
                print(f"❌ 处理失败: {img_path} - {e}")
        
        total_time = time.time() - start_time
        
        # 统计摘要
        self._print_batch_summary(results, total_time)
        
        return results
    
    def predict_directory(self, directory, pattern="*.jpg", recursive=False):
        """
        识别目录下所有图片
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            recursive: 是否递归子目录
        """
        directory = Path(directory)
        
        if not directory.exists():
            print(f"❌ 目录不存在: {directory}")
            return None
        
        # 搜索图片 - 支持多种常见格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp', '*.BMP']
        image_paths = []
        
        for ext in image_extensions:
            if recursive:
                image_paths.extend(list(directory.rglob(ext)))
            else:
                image_paths.extend(list(directory.glob(ext)))
        
        # 去重（防止大小写重复）
        image_paths = list(set(image_paths))
        
        if len(image_paths) == 0:
            print(f"❌ 未找到图片文件: {directory}")
            print(f"💡 提示: 支持的格式: jpg, jpeg, png, bmp")
            return None
        
        print(f"\n📁 目录: {directory}")
        print(f"🔍 模式: {pattern}")
        print(f"📊 找到 {len(image_paths)} 张图片")
        
        return self.predict_batch(image_paths)
    
    def interactive_mode(self):
        """交互式命令行模式"""
        print("\n" + "="*60)
        print("🎮 进入交互式模式")
        print("="*60)
        print("\n命令说明:")
        print("  <图片路径>        - 识别单张图片")
        print("  <目录路径>        - 识别目录下所有图片 (自动检测)")
        print("  dir <目录>        - 识别目录下所有图片")
        print("  batch <文件1> <文件2> ... - 批量识别多张图片")
        print("  history           - 查看识别历史")
        print("  stats             - 显示统计信息")
        print("  clear             - 清除历史")
        print("  help              - 显示帮助")
        print("  quit/exit         - 退出程序")
        print("="*60 + "\n")
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n🎯 请输入命令 > ").strip()
                
                if not user_input:
                    continue
                
                # 解析命令
                parts = user_input.split()
                command = parts[0].lower()
                
                # 处理命令
                if command in ['quit', 'exit', 'q']:
                    print("\n👋 感谢使用宝可梦图鉴! Bye~")
                    break
                
                elif command == 'help':
                    self._print_help()
                
                elif command == 'history':
                    self._print_history()
                
                elif command == 'stats':
                    self._print_stats()
                
                elif command == 'clear':
                    self.history.clear()
                    print("✅ 历史记录已清除")
                
                elif command == 'dir':
                    if len(parts) < 2:
                        print("❌ 用法: dir <目录路径>")
                    else:
                        self.predict_directory(parts[1])
                
                elif command == 'batch':
                    if len(parts) < 2:
                        print("❌ 用法: batch <图片1> <图片2> ...")
                    else:
                        self.predict_batch(parts[1:])
                
                else:
                    # 智能判断：目录 or 文件
                    input_path = Path(user_input.strip())
                    
                    if input_path.exists():
                        if input_path.is_dir():
                            # 自动识别为目录
                            print(f"💡 检测到目录，自动切换到目录识别模式")
                            self.predict_directory(user_input)
                        elif input_path.is_file():
                            # 单张图片
                            self.predict_single(user_input, show_image=True)
                        else:
                            print(f"❌ 不支持的路径类型: {user_input}")
                    else:
                        print(f"❌ 路径不存在: {user_input}")
                        print("💡 提示: 请检查路径是否正确，或使用 'help' 查看命令帮助")
            
            except KeyboardInterrupt:
                print("\n\n⚠️  接收到中断信号")
                confirm = input("确定要退出吗? (y/n) > ").strip().lower()
                if confirm in ['y', 'yes']:
                    break
            
            except Exception as e:
                print(f"❌ 错误: {e}")
                import traceback
                traceback.print_exc()
    
    def _add_text_to_image(self, img, result):
        """在图片上添加识别结果"""
        h, w = img.shape[:2]
        
        # 创建副本
        img_display = img.copy()
        
        # 调整图片大小以便显示
        max_size = 800
        if w > max_size or h > max_size:
            scale = min(max_size/w, max_size/h)
            new_w, new_h = int(w*scale), int(h*scale)
            img_display = cv2.resize(img_display, (new_w, new_h))
            h, w = new_h, new_w
        
        # 添加黑色背景
        overlay = img_display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img_display, 0.3, 0, img_display)
        
        # 添加文字
        pokemon_name = result.get('top1_name', f"ID:{result['top1_label']}")
        conf = result['top1_conf']
        
        cv2.putText(img_display, f"Pokemon: {pokemon_name}", 
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(img_display, f"Confidence: {conf:.2%}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return img_display
    
    def _print_batch_summary(self, results, total_time):
        """打印批量识别摘要"""
        print(f"\n{'='*60}")
        print(f"✅ 批量识别完成!")
        print(f"{'='*60}")
        print(f"总计: {len(results)} 张图片")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均速度: {total_time/len(results)*1000:.2f} ms/张")
        
        # 统计置信度分布
        if results:
            confidences = [r['result']['top1_conf'] for r in results]
            print(f"\n置信度统计:")
            print(f"  最高: {max(confidences):.4f}")
            print(f"  最低: {min(confidences):.4f}")
            print(f"  平均: {np.mean(confidences):.4f}")
            
            # 高置信度预测
            high_conf = [r for r in results if r['result']['top1_conf'] > 0.9]
            print(f"  高置信度(>0.9): {len(high_conf)}/{len(results)}")
    
    def _print_history(self):
        """打印识别历史"""
        if not self.history:
            print("📭 暂无识别历史")
            return
        
        print(f"\n{'='*60}")
        print(f"📜 识别历史 (共 {len(self.history)} 条)")
        print(f"{'='*60}")
        
        for i, record in enumerate(self.history[-10:], 1):  # 只显示最近10条
            result = record['result']
            pokemon_name = result.get('top1_name', f"ID:{result['top1_label']}")
            conf = result['top1_conf']
            filename = Path(record['file']).name
            
            print(f"{i:2d}. {filename:40s} -> {pokemon_name:20s} ({conf:.4f})")
        
        if len(self.history) > 10:
            print(f"\n... 还有 {len(self.history)-10} 条历史记录")
    
    def _print_stats(self):
        """打印统计信息"""
        if not self.history:
            print("📭 暂无统计数据")
            return
        
        print(f"\n{'='*60}")
        print(f"📊 统计信息")
        print(f"{'='*60}")
        
        print(f"总识别次数: {len(self.history)}")
        
        # 统计最常识别的宝可梦
        from collections import Counter
        predictions = [r['result'].get('top1_name', 'Unknown') for r in self.history]
        most_common = Counter(predictions).most_common(5)
        
        print(f"\n最常识别的宝可梦:")
        for i, (pokemon, count) in enumerate(most_common, 1):
            print(f"  {i}. {pokemon:20s} - {count} 次")
        
        # 平均置信度
        confidences = [r['result']['top1_conf'] for r in self.history]
        print(f"\n平均置信度: {np.mean(confidences):.4f}")
    
    def _print_help(self):
        """打印帮助信息"""
        print("\n" + "="*60)
        print("📖 命令帮助")
        print("="*60)
        print("\n基本命令:")
        print("  <图片路径>        识别单张图片")
        print("                    示例: test.jpg")
        print("                    示例: /home/pi/pokemon/pikachu.png")
        print("")
        print("  <目录路径>        识别目录下所有图片 (自动检测)")
        print("                    示例: /home/pi/test_images")
        print("                    示例: ./test_random/")
        print("")
        print("  dir <目录>        识别目录下所有图片")
        print("                    示例: dir /home/pi/test_images")
        print("")
        print("  batch <文件列表>  批量识别多张图片")
        print("                    示例: batch img1.jpg img2.jpg img3.jpg")
        print("")
        print("查询命令:")
        print("  history           查看识别历史")
        print("  stats             显示统计信息")
        print("  clear             清除历史记录")
        print("")
        print("系统命令:")
        print("  help              显示此帮助信息")
        print("  quit/exit         退出程序")
        print("="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="宝可梦图鉴 - 树莓派交互式识别系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 交互式模式
  python deploy_interactive.py
  
  # 识别单张图片
  python deploy_interactive.py -i test.jpg
  
  # 识别目录
  python deploy_interactive.py -d /home/pi/test_images
  
  # 批量识别
  python deploy_interactive.py -b img1.jpg img2.jpg img3.jpg
        """
    )
    
    parser.add_argument('-m', '--model', type=str,
                       default='~/pokemon/best.onnx',
                       help='ONNX模型路径')
    
    parser.add_argument('-n', '--names', type=str,
                       default='pokemon_names.json',
                       help='类别名称文件路径')
    
    parser.add_argument('-t', '--threshold', type=float,
                       default=0.5,
                       help='置信度阈值 (默认: 0.5)')
    
    parser.add_argument('-i', '--image', type=str,
                       help='单张图片路径')
    
    parser.add_argument('-d', '--directory', type=str,
                       help='图片目录路径')
    
    parser.add_argument('-b', '--batch', nargs='+',
                       help='批量图片路径列表')
    
    parser.add_argument('--show', action='store_true',
                       help='显示图片(需要图形界面)')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not Path(args.model).exists():
        print(f"❌ 模型文件不存在: {args.model}")
        print("\n提示:")
        print("  1. 请先训练模型并导出ONNX格式")
        print("  2. 或使用 -m 参数指定正确的模型路径")
        return
    
    # 初始化
    try:
        app = InteractivePokedex(
            model_path=args.model,
            names_file=args.names if Path(args.names).exists() else None,
            conf_threshold=args.threshold
        )
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 根据参数执行不同模式
    if args.image:
        # 单图模式
        app.predict_single(args.image, show_image=args.show)
    
    elif args.directory:
        # 目录模式
        app.predict_directory(args.directory)
    
    elif args.batch:
        # 批量模式
        app.predict_batch(args.batch)
    
    else:
        # 交互式模式
        app.interactive_mode()

if __name__ == "__main__":
    main()

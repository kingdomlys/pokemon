"""
树莓派4B 宝可梦图鉴部署脚本
使用ONNX Runtime进行推理,性能优化版
支持Google TTS中文语音播报功能
"""
import cv2
import numpy as np
import onnxruntime as ort
import time
from pathlib import Path
import json
import os
import tempfile
import subprocess

# Google TTS 语音支持（可选）
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  gtts 未安装，语音播报功能不可用")
    print("   安装命令: pip install gtts")

class PokemonPokedex:
    """宝可梦图鉴识别器"""
    
    def __init__(self, model_path, names_file=None, conf_threshold=0.5, enable_tts=True):
        """
        初始化图鉴
        
        Args:
            model_path: ONNX模型路径
            names_file: 类别名称文件(JSON格式)
            conf_threshold: 置信度阈值
            enable_tts: 是否启用语音播报
        """
        print("🎮 初始化宝可梦图鉴...")
        
        # 初始化 Google TTS
        self.tts_enabled = enable_tts and TTS_AVAILABLE
        self.temp_dir = tempfile.gettempdir()
        
        if self.tts_enabled:
            try:
                # 测试网络连接和 gtts
                print("🔊 初始化 Google TTS...")
                
                # 检查音频播放工具
                self.audio_player = self._detect_audio_player()
                if not self.audio_player:
                    print("⚠️  未找到音频播放工具 (mpg123/ffplay)")
                    print("   安装: sudo apt-get install mpg123")
                    self.tts_enabled = False
                else:
                    print(f"✅ 语音播报已启用 (使用 {self.audio_player})")
                    
            except Exception as e:
                print(f"⚠️  TTS初始化失败: {e}")
                self.tts_enabled = False
        else:
            if enable_tts and not TTS_AVAILABLE:
                print("💡 提示: 安装 gtts 以启用中文语音播报")
                print("   pip install gtts")
                print("   sudo apt-get install mpg123")
        
        # 加载ONNX模型
        print(f"📦 加载模型: {model_path}")
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']  # 树莓派使用CPU
        )
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        print(f"   输入名称: {self.input_name}")
        print(f"   输入形状: {self.input_shape}")
        print(f"   输出名称: {self.output_name}")
        
        # 加载类别名称和详细信息
        if names_file and Path(names_file).exists():
            with open(names_file, 'r', encoding='utf-8') as f:
                raw_names = json.load(f)
            
            # 保存原始详细信息（用于显示）
            self.pokemon_details = raw_names
            
            # 标准化名称映射（用于快速查找）
            self.names = self._normalize_names(raw_names)
            if self.names:
                print(f"   加载 {len(self.pokemon_details)} 个宝可梦类别（含详细信息）")
            else:
                print("   警告: 类别名称文件为空或格式不兼容，使用默认编号")
        else:
            self.names = None
            self.pokemon_details = {}
            print("   警告: 未提供类别名称文件")
        
        self.conf_threshold = conf_threshold
        self.img_size = 224  # YOLOv8-cls默认输入大小
        
        print("✅ 图鉴初始化完成!\n")
    
    def _detect_audio_player(self):
        """检测可用的音频播放工具"""
        # 优先使用 mpg123
        try:
            result = subprocess.run(['mpg123', '--version'], 
                                  capture_output=True, timeout=2)
            if result.returncode == 0:
                return 'mpg123'
        except:
            pass
        
        # 备用 ffplay
        try:
            result = subprocess.run(['ffplay', '-version'], 
                                  capture_output=True, timeout=2)
            if result.returncode == 0:
                return 'ffplay'
        except:
            pass
        
        return None
    
    def preprocess(self, image):
        """
        图像预处理
        
        Args:
            image: OpenCV读取的图像(BGR格式)
        
        Returns:
            预处理后的张量
        """
        # 调整大小
        img = cv2.resize(image, (self.img_size, self.img_size))
        
        # BGR转RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 归一化到[0, 1]
        img = img.astype(np.float32) / 255.0
        
        # 转换为CHW格式
        img = np.transpose(img, (2, 0, 1))
        
        # 添加batch维度
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def postprocess(self, output):
        """
        后处理输出
        
        Args:
            output: 模型输出
        
        Returns:
            预测结果字典
        """
        # Flatten logits before softmax so class index selection works
        logits = np.asarray(output[0])
        probs = self._softmax(np.squeeze(logits))
        
        # Top-1
        top1_idx = np.argmax(probs)
        top1_conf = probs[top1_idx]
        
        # Top-5
        top5_idx = np.argsort(probs)[::-1][:5]
        top5_conf = probs[top5_idx]
        
        # 类别索引从0开始，但数据集编号从0001开始，需要+1对齐
        # 例如：模型输出0 -> 0001妙蛙种子，模型输出385 -> 0386
        top1_label_aligned = int(top1_idx) + 1
        top5_labels_aligned = [int(i) + 1 for i in top5_idx]
        
        result = {
            'top1_label': top1_label_aligned,
            'top1_conf': float(top1_conf)*100,
            'top5_labels': top5_labels_aligned,
            'top5_conf': [float(c)*100 for c in top5_conf]
        }
        
        # 添加名称
        if self.names:
            result['top1_name'] = self._resolve_name(top1_label_aligned)
            result['top5_names'] = [self._resolve_name(i) for i in top5_labels_aligned]
        
        return result

    def _normalize_names(self, raw_names):
        """标准化名称映射，兼容列表、数字字符串等格式"""
        if raw_names is None:
            return {}

        normalized = {}

        if isinstance(raw_names, list):
            for idx, name in enumerate(raw_names):
                if not name:
                    continue
                # 如果是字符串直接用，如果是字典则提取 name 字段
                display_name = name.get('name', f'Pokemon_{idx}') if isinstance(name, dict) else name
                normalized[str(idx)] = display_name
                normalized[f"{idx:04d}"] = display_name
        elif isinstance(raw_names, dict):
            for key, value in raw_names.items():
                if not value:
                    continue
                str_key = str(key)
                
                # 如果 value 是字典（包含详细信息），提取 name 字段
                if isinstance(value, dict):
                    display_name = value.get('name', f'Pokemon_{key}')
                else:
                    display_name = value
                
                if str_key.isdigit():
                    idx = int(str_key)
                    normalized[str(idx)] = display_name
                    normalized[f"{idx:04d}"] = display_name
                normalized[str_key] = display_name

        return normalized

    def _resolve_name(self, class_idx):
        """根据类别索引返回名称，找不到则返回 Unknown_x"""
        if not self.names:
            return f"Unknown_{class_idx}"

        key_plain = str(class_idx)
        key_zero = f"{class_idx:04d}"
        return self.names.get(key_plain) or self.names.get(key_zero) or f"Unknown_{class_idx}"
    
    def _get_pokemon_details(self, class_idx):
        """根据类别索引获取宝可梦的详细信息"""
        if not self.pokemon_details:
            return None
        
        key_zero = f"{class_idx:04d}"
        return self.pokemon_details.get(key_zero, None)
    
    def _format_pokemon_info(self, details):
        """格式化宝可梦详细信息为一段话"""
        if not details or not isinstance(details, dict):
            return ""
        
        info_parts = []
        
        # 基本信息
        name_cn = details.get('name_cn', '')
        name_en = details.get('name_en', '')
        category = details.get('category', '')
        
        if name_cn and category:
            info_parts.append(f"{name_cn}, {category}")
        
        # 属性
        types = details.get('types', [])
        if types:
            types_str = "、".join(types)
            info_parts.append(f"属性为{types_str}系")
        
        # 特性
        abilities = details.get('abilities', [])
        if abilities:
            abilities_str = "、".join(abilities)
            info_parts.append(f"拥有{abilities_str}等特性")
        
        # 体型
        height = details.get('height', '')
        weight = details.get('weight', '')
        if height and weight:
            info_parts.append(f"身高{height}，体重{weight}")
        
        # 种族值
        stats = details.get('stats', {})
        if stats and isinstance(stats, dict):
            total = stats.get('total', '')
            if total:
                hp = stats.get('hp', '')
                attack = stats.get('attack', '')
                defense = stats.get('defense', '')
                info_parts.append(f"种族值总和{total}（HP:{hp} 攻击:{attack} 防御:{defense}）")
        
        # 拼接成一段话
        if info_parts:
            return "，".join(info_parts) + "。"
        return ""
    
    def _speak(self, text):
        """
        使用 Google TTS 播报文本
        
        Args:
            text: 要播报的中文文本
        """
        if not self.tts_enabled:
            return
        
        try:
            print(f"🔊 播报中...")
            
            # 生成临时音频文件
            audio_file = os.path.join(self.temp_dir, 'pokemon_tts_temp.mp3')
            
            # 使用 Google TTS 生成音频
            tts = gTTS(text=text, lang='zh-cn', slow=False)
            tts.save(audio_file)
            
            # 播放音频
            if self.audio_player == 'mpg123':
                subprocess.run(['mpg123', '-q', audio_file], 
                             timeout=30, 
                             stderr=subprocess.DEVNULL)
            elif self.audio_player == 'ffplay':
                subprocess.run(['ffplay', '-nodisp', '-autoexit', audio_file], 
                             timeout=30, 
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            
            # 清理临时文件
            if os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                except:
                    pass  # 忽略删除失败
                    
        except Exception as e:
            print(f"⚠️  语音播报失败: {e}")
            # 如果是网络问题，提示用户
            if "Connection" in str(e) or "Network" in str(e):
                print("   提示: 请检查网络连接（Google TTS 需要网络）")
    
    def _softmax(self, x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def predict(self, image, verbose=True):
        """
        预测图像
        
        Args:
            image: 输入图像或图像路径
            verbose: 是否打印结果
        
        Returns:
            预测结果字典
        """
        # 读取图像
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"无法读取图像: {image}")
        
        # 预处理
        input_tensor = self.preprocess(image)
        
        # 推理
        start_time = time.time()
        output = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        inference_time = (time.time() - start_time) * 1000  # 毫秒
        
        # 后处理
        result = self.postprocess(output)
        result['inference_time'] = inference_time
        
        # 打印结果
        if verbose:
            self._print_result(result)
        
        return result
    
    def _print_result(self, result):
        """打印预测结果"""
        print("\n" + "="*60)
        if 'top1_name' in result:
            print(f"🎯 识别到宝可梦: {result['top1_name']}")
        else:
            print(f"🎯 预测类别: {result['top1_label']}")
        
        print(f"   置信度: {result['top1_conf']:.4f}")
        print(f"   推理时间: {result['inference_time']:.2f} ms")
        
        if result['top1_conf'] < self.conf_threshold:
            print(f"   ⚠️  置信度低于阈值 {self.conf_threshold}")
        
        # 显示详细信息
        top1_label = result.get('top1_label', 0)
        details = self._get_pokemon_details(top1_label)
        info_text = ""
        
        if details:
            info_text = self._format_pokemon_info(details)
            if info_text:
                print(f"\n📖 宝可梦图鉴:")
                print(f"   {info_text}")
        
        # 语音播报
        if self.tts_enabled and info_text:
            # 播报宝可梦名称和详细信息
            pokemon_name = details.get('name_cn', '')
            if pokemon_name:
                tts_text = f"识别到{pokemon_name}。{info_text}"
            else:
                tts_text = info_text
            
            self._speak(tts_text)
        
        print(f"\n📊 Top-5 预测:")
        for i, (label, conf) in enumerate(
            zip(result['top5_labels'], result['top5_conf']), 1
        ):
            if 'top5_names' in result:
                name = result['top5_names'][i-1]
                print(f"   {i}. {name:20s} - {conf:.4f}")
            else:
                print(f"   {i}. Label {label:3d} - {conf:.4f}")
        print("="*60)
    
    def benchmark(self, image, n_runs=100):
        """
        性能基准测试
        
        Args:
            image: 测试图像
            n_runs: 运行次数
        """
        print(f"\n🔧 运行性能测试 ({n_runs} 次推理)...")
        
        # 读取和预处理
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        input_tensor = self.preprocess(image)
        
        # 预热
        for _ in range(10):
            self.session.run([self.output_name], {self.input_name: input_tensor})
        
        # 测试
        times = []
        for _ in range(n_runs):
            start = time.time()
            self.session.run([self.output_name], {self.input_name: input_tensor})
            times.append((time.time() - start) * 1000)
        
        # 统计
        times = np.array(times)
        print(f"\n性能统计:")
        print(f"  平均推理时间: {times.mean():.2f} ms")
        print(f"  最小推理时间: {times.min():.2f} ms")
        print(f"  最大推理时间: {times.max():.2f} ms")
        print(f"  标准差: {times.std():.2f} ms")
        print(f"  平均FPS: {1000/times.mean():.2f}")

def create_names_file_from_pytorch(pt_model_path, output_path="pokemon_names.json"):
    """
    从PyTorch模型提取类别名称并保存为JSON
    
    Args:
        pt_model_path: .pt模型路径
        output_path: 输出JSON文件路径
    """
    try:
        from ultralytics import YOLO
        print(f"📝 从 {pt_model_path} 提取类别名称...")
        
        model = YOLO(pt_model_path)
        names = model.names
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(names, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 类别名称已保存到: {output_path}")
        print(f"   共 {len(names)} 个类别")
        
    except ImportError:
        print("❌ 需要安装ultralytics库才能提取类别名称")
        print("   在有ultralytics的环境中运行此函数")

def main():
    """主函数 - 演示用法"""
    
    # ===== 配置 =====
    MODEL_PATH = "runs/classify/pokemon_yolov8n/weights/best.onnx"
    NAMES_FILE = "pokemon_names.json"
    TEST_IMAGE = "Dataset_pokemon/0001/0001Bulbasaur1.jpg"
    
    # ===== 创建类别名称文件(仅需运行一次) =====
    if not Path(NAMES_FILE).exists():
        print("⚠️  类别名称文件不存在,尝试从.pt模型提取...")
        pt_model = "runs/classify/pokemon_yolov8n/weights/best.pt"
        if Path(pt_model).exists():
            create_names_file_from_pytorch(pt_model, NAMES_FILE)
    
    # ===== 初始化图鉴 =====
    pokedex = PokemonPokedex(
        model_path=MODEL_PATH,
        names_file=NAMES_FILE,
        conf_threshold=0.5
    )
    
    # ===== 测试单张图片 =====
    if Path(TEST_IMAGE).exists():
        print(f"\n📸 测试图片: {TEST_IMAGE}")
        result = pokedex.predict(TEST_IMAGE)
    else:
        print(f"⚠️  测试图片不存在: {TEST_IMAGE}")
    
    # ===== 性能测试 =====
    if Path(TEST_IMAGE).exists():
        pokedex.benchmark(TEST_IMAGE, n_runs=100)
    
    print("\n🎉 演示完成!")
    print("\n💡 提示: 在树莓派上使用摄像头实时识别:")
    print("   1. 连接USB摄像头或树莓派摄像头模块")
    print("   2. 运行: python deploy_realtime.py")

if __name__ == "__main__":
    main()

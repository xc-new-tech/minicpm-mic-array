# MiniCPM-o 2.6 + 麦克风阵列 嘈杂环境语音对话系统

基于 MiniCPM-o 2.6 多模态模型，结合麦克风阵列前端处理和 Demucs 人声分离技术，实现嘈杂环境下的语音识别和对话能力。

## 应用场景

商场美妆集合店数字人产品 - 在嘈杂的商场环境中准确识别顾客语音并进行对话。

## 功能特性

### 麦克风阵列处理
- **声源定位 (DOA)** - SRP-PHAT 算法，定位误差 < 6°
- **多声源定位** - 同时检测多个声源方向
- **波束成形** - Delay-and-Sum 算法，定向拾音
- **语音活动检测 (VAD)** - WebRTC VAD
- **降噪处理** - 谱减法

### 人声检测与分离
- **人声检测** - F0 基频 + VAD + 频谱特征综合判断
- **人声分离** - Demucs (htdemucs) 深度学习模型
- **人声方向定位** - 只聚焦人声方向，过滤音乐/噪声

### 语音识别
- **MiniCPM-o 2.6** - 多模态大模型，支持语音识别和对话
- **Apple Silicon 优化** - 支持 MPS 加速

## 系统要求

- Python 3.10
- Apple M4 Max + 128GB RAM（推荐）或 CUDA GPU
- 约 16GB 存储空间（模型文件）

## 安装

```bash
# 创建虚拟环境
conda create -n minicpm python=3.10
conda activate minicpm

# 安装依赖
pip install -r requirements.txt

# 模型会在首次运行时自动下载
```

## 使用方法

### 基础测试
```bash
# 运行基础语音测试
python test_audio.py

# 运行麦克风阵列完整测试
python test_with_mic_array.py
```

### 代码示例

```python
from mic_array import MicArrayProcessor, create_linear_array

# 创建 4 麦克风线性阵列
mic_positions = create_linear_array(n_mics=4, spacing=0.05)
processor = MicArrayProcessor(mic_positions, sample_rate=16000)

# 处理多通道音频（启用人声分离）
result = processor.process(
    multi_channel_audio,
    auto_beamform=True,
    denoise=True,
    human_voice_only=True,
    use_separation=True
)

# 获取处理后的音频和声源方向
enhanced_audio = result['audio']
doa_info = result['doa']
```

## 测试结果

### 声源定位精度
| 真实方向 | 检测方向 | 误差 |
|---------|---------|------|
| 30° | 33.6° | 3.6° |
| 45° | 50.6° | 5.6° |

### 语音识别对比 (预期: "你好，请推荐一款适合干皮的粉底液")

| 噪声级别 | 原始录音 | 波束成形 | 人声分离(Demucs) |
|---------|---------|---------|-----------------|
| 0.05 (轻度) | ❌ | ❌ | ✅ 有效 |
| 0.10 (中度) | ❌ | ❌ | ❌ |
| 0.15 (重度) | ❌ | ❌ | ❌ |

### 人声检测置信度提升

| 方法 | F0检测 | VAD比例 | 总置信度 |
|------|--------|--------|---------|
| 无人声分离 | 0.00 | 0.99 | 0.47 |
| 有人声分离 | 0.66 | 0.93 | 0.73 |

## 处理流程

```
多通道音频输入
      │
      ▼
┌─────────────────┐
│ 多声源定位     │  ← SRP-PHAT
└────────┬────────┘
         │
    ┌────┴────┬────────┐
    ▼         ▼        ▼
┌───────┐ ┌───────┐ ┌───────┐
│波束成形│ │波束成形│ │波束成形│
└───┬───┘ └───┬───┘ └───┬───┘
    │         │        │
    ▼         ▼        ▼
┌───────┐ ┌───────┐ ┌───────┐
│Demucs │ │Demucs │ │Demucs │  ← 人声分离
└───┬───┘ └───┬───┘ └───┬───┘
    │         │        │
    ▼         ▼        ▼
┌───────┐ ┌───────┐ ┌───────┐
│人声检测│ │人声检测│ │人声检测│  ← F0+VAD+频谱
└───┬───┘ └───┬───┘ └───┬───┘
    │         │        │
    └────┬────┴────────┘
         │
         ▼
┌─────────────────┐
│ 选择置信度最高  │
│   的人声方向    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MiniCPM-o 2.6 │
│   ASR / 对话    │
└─────────────────┘
```

## 文件结构

```
minicpm/
├── mic_array.py              # 麦克风阵列处理模块
├── test_audio.py             # 基础音频测试
├── test_with_mic_array.py    # 完整集成测试
├── test_input.wav            # 测试音频
├── requirements.txt          # 依赖列表
├── outputs/                  # 测试输出目录
└── todo.md                   # 项目文档
```

## 结论与建议

### 优点
1. 声源定位准确，误差 < 6°
2. 人声分离在轻度噪声下显著提升识别效果
3. F0 + VAD + 频谱的人声检测方案准确可靠

### 局限
1. MiniCPM-o 对噪声敏感，噪声级别 ≥ 0.1 时识别能力下降
2. Demucs 模型计算量较大

### 建议
1. 控制环境噪声 SNR > 15dB
2. 麦克风尽量靠近用户
3. 考虑使用专用 ASR 模型（如 Whisper）处理高噪声场景
4. 增加麦克风数量（6-8个）可提升效果

参考开源：https://github.com/modelscope/ClearerVoice-Studio


## License

MIT

## 参考

- [MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o) - 多模态大模型
- [Demucs](https://github.com/facebookresearch/demucs) - 音乐/人声分离
- [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) - 房间声学模拟

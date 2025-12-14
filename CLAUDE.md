# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

MiniCPM-o 2.6 + 麦克风阵列前端处理测试项目，用于评估多模态AI模型在嘈杂环境下的语音识别和对话能力。

**应用场景**: 商场美妆集合店数字人产品
**硬件要求**: Apple M4 Max + 128GB RAM + 4麦克风阵列

## 常用命令

```bash
# 激活虚拟环境 (conda)
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /Users/xc-tech/code/minicpm/venv

# 运行基础语音测试 (不同噪声级别下的ASR)
python test_audio.py

# 运行完整麦克风阵列测试 (DOA + 波束成形 + 降噪 + ASR)
python test_with_mic_array.py

# 单独测试麦克风阵列前端处理模块
python mic_array.py
```

## 代码架构

### 核心模块

- **mic_array.py** - 麦克风阵列前端处理模块
  - `MicArrayProcessor` - 主处理器类，封装完整处理流程
  - `estimate_doa()` - SRP-PHAT算法单声源定位
  - `estimate_doa_multi()` - 多声源定位（返回多个候选方向）
  - `beamform()` - Delay-and-Sum波束成形
  - `noise_reduce()` - 谱减法降噪
  - `detect_voice_activity()` - WebRTC VAD语音检测
  - `is_human_voice()` - 人声检测（F0基频+VAD+频谱特征）
  - `separate_voice()` - Demucs人声分离
  - `find_human_voice_direction()` - 人声方向定位（过滤音乐/噪声）
  - `simulate_mic_array_recording()` - 使用pyroomacoustics模拟多通道录音

- **test_audio.py** - 基础语音测试脚本
  - 测试MiniCPM-o在不同信噪比下的ASR性能
  - 包含`flash_attn`模块mock绕过(macOS不支持)

- **test_with_mic_array.py** - 完整集成测试
  - 组合麦克风阵列处理 + 人声分离 + MiniCPM-o ASR
  - 对比：原始录音 vs 波束成形 vs 人声分离

### 处理流程

**传统流程:**
```
多通道音频 → DOA声源定位 → 波束成形 → 降噪 → MiniCPM-o ASR
```

**人声分离增强流程 (推荐):**
```
多通道音频 → 多声源定位(3个候选) → 各方向波束成形 → Demucs人声分离
         → 人声检测(F0+VAD) → 选择人声置信度最高的方向 → 降噪 → ASR
```

### 关键依赖

- `pyroomacoustics` - 声学仿真和阵列处理
- `webrtcvad` - 语音活动检测
- `demucs` - 人声分离（htdemucs模型，~80MB）
- `librosa` - 音频分析（F0检测等）
- `transformers==4.44.2` - MiniCPM-o模型加载
- `torch` (bfloat16 on MPS) - 推理

## 使用示例

### 基础处理（传统波束成形）
```python
from mic_array import MicArrayProcessor, create_linear_array

mic_positions = create_linear_array(n_mics=4, spacing=0.05)
processor = MicArrayProcessor(mic_positions, sample_rate=16000)

result = processor.process(
    multi_channel_audio,
    auto_beamform=True,
    denoise=True
)
# result['audio'] - 处理后音频
# result['doa']['azimuth'] - 声源方向
```

### 人声定位 + 人声分离（推荐）
```python
result = processor.process(
    multi_channel_audio,
    auto_beamform=True,
    denoise=True,
    human_voice_only=True,      # 启用人声定位
    use_separation=True,        # 启用Demucs人声分离
    num_candidates=3            # 候选声源数量
)
# result['human_voice_detection'] - 人声检测详情
```

### 单独使用人声分离
```python
# 提取纯净人声
vocals = processor.separate_voice(audio)

# 获取所有音轨
tracks = processor.separate_voice(audio, return_all=True)
# tracks: {'vocals', 'drums', 'bass', 'other'}
```

## 注意事项

- 模型使用`sdpa` attention实现(替代flash_attn，因macOS不支持)
- 虚拟环境为conda环境，路径在`/Users/xc-tech/code/minicpm/venv`
- 输出文件保存在`outputs/`目录
- Demucs模型首次运行会自动下载(~80MB)，使用MPS加速
- MiniCPM-o对噪声敏感，建议SNR > 15dB (噪声级别 < 0.1)

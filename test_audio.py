#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MiniCPM-o 2.6 嘈杂环境语音测试脚本

应用场景：商场美妆集合店数字人产品
测试目标：评估模型在嘈杂环境下的语音识别和对话能力
"""

import os
import torch
import librosa
import soundfile as sf
import numpy as np
from transformers import AutoModel, AutoTokenizer


def get_device():
    """获取可用设备"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model():
    """加载 MiniCPM-o 2.6 模型"""
    print("正在加载模型...")
    device = get_device()
    print(f"使用设备: {device}")

    # 创建假的 flash_attn 模块以绕过 import 检查
    import sys
    import types
    from importlib.machinery import ModuleSpec

    fake_flash_attn = types.ModuleType('flash_attn')
    fake_flash_attn.__spec__ = ModuleSpec('flash_attn', None)
    fake_flash_attn.__version__ = "0.0.0"
    fake_flash_attn.flash_attn_func = None
    fake_flash_attn.flash_attn_varlen_func = None
    sys.modules['flash_attn'] = fake_flash_attn

    fake_interface = types.ModuleType('flash_attn.flash_attn_interface')
    fake_interface.__spec__ = ModuleSpec('flash_attn.flash_attn_interface', None)
    sys.modules['flash_attn.flash_attn_interface'] = fake_interface

    model = AutoModel.from_pretrained(
        'openbmb/MiniCPM-o-2_6',
        trust_remote_code=True,
        attn_implementation='sdpa',  # 使用 SDPA 而不是 flash_attention
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        init_vision=True,
        init_audio=True,
        init_tts=True
    )

    if device == "mps":
        model = model.to(device)
    elif device == "cuda":
        model = model.cuda()

    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)

    # 初始化 TTS
    model.init_tts()
    model.tts.float()

    print("模型加载完成!")
    return model, tokenizer


def add_noise(audio, noise_level=0.1):
    """添加白噪声模拟嘈杂环境"""
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise


def add_crowd_noise(audio, crowd_audio_path=None, snr_db=10):
    """
    添加人群噪声模拟商场环境
    snr_db: 信噪比(dB)，越小越嘈杂
    """
    if crowd_audio_path and os.path.exists(crowd_audio_path):
        crowd, _ = librosa.load(crowd_audio_path, sr=16000, mono=True)
        # 确保噪声长度与音频匹配
        if len(crowd) < len(audio):
            crowd = np.tile(crowd, int(np.ceil(len(audio) / len(crowd))))
        crowd = crowd[:len(audio)]
    else:
        # 生成模拟商场噪声（混合多种频率）
        t = np.arange(len(audio)) / 16000
        crowd = (
            np.random.randn(len(audio)) * 0.3 +  # 白噪声底噪
            np.sin(2 * np.pi * 200 * t) * 0.1 +  # 低频嗡嗡声
            np.sin(2 * np.pi * 500 * t) * 0.05   # 中频
        )

    # 计算信噪比
    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(crowd ** 2)

    if noise_power > 0:
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        crowd = crowd * np.sqrt(target_noise_power / noise_power)

    return audio + crowd


def test_speech_recognition(model, tokenizer, audio_path, output_dir="outputs"):
    """测试语音识别能力"""
    os.makedirs(output_dir, exist_ok=True)

    # 加载音频
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    print(f"音频长度: {len(audio)/16000:.2f}秒")

    # 测试不同噪声级别
    noise_levels = [
        ("clean", audio, "无噪声"),
        ("light", add_crowd_noise(audio, snr_db=20), "轻度噪声 (SNR=20dB)"),
        ("moderate", add_crowd_noise(audio, snr_db=10), "中度噪声 (SNR=10dB)"),
        ("heavy", add_crowd_noise(audio, snr_db=5), "重度噪声 (SNR=5dB)"),
        ("extreme", add_crowd_noise(audio, snr_db=0), "极端噪声 (SNR=0dB)"),
    ]

    results = []

    for name, noisy_audio, desc in noise_levels:
        print(f"\n--- 测试: {desc} ---")

        # 保存带噪音频以供检查
        noisy_path = os.path.join(output_dir, f"noisy_{name}.wav")
        sf.write(noisy_path, noisy_audio, 16000)

        # 构建消息
        task_prompt = "请仔细听这段音频并转录内容。\n"
        msgs = [{'role': 'user', 'content': [task_prompt, noisy_audio]}]

        try:
            res = model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                max_new_tokens=256,
                use_tts_template=False,
                generate_audio=False,
                temperature=0.3
            )
            print(f"识别结果: {res}")
            results.append((name, desc, res))
        except Exception as e:
            print(f"识别失败: {e}")
            results.append((name, desc, f"ERROR: {e}"))

    return results


def test_dialogue(model, tokenizer, audio_path, output_dir="outputs"):
    """测试对话能力"""
    os.makedirs(output_dir, exist_ok=True)

    # 加载音频
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # 添加中度噪声模拟商场环境
    noisy_audio = add_crowd_noise(audio, snr_db=10)

    print("\n--- 测试对话能力 (商场环境 SNR=10dB) ---")

    # 使用助手模式
    sys_prompt = model.get_sys_prompt(mode='audio_assistant', language='zh')

    msgs = [
        sys_prompt,
        {'role': 'user', 'content': [noisy_audio]}
    ]

    try:
        output_path = os.path.join(output_dir, "response.wav")
        res = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            max_new_tokens=256,
            use_tts_template=True,
            generate_audio=True,
            temperature=0.3,
            output_audio_path=output_path
        )
        print(f"对话响应: {res}")
        print(f"音频输出: {output_path}")
        return res
    except Exception as e:
        print(f"对话失败: {e}")
        return None


def create_test_audio(output_path="test_input.wav"):
    """创建测试音频（需要手动录制或使用现有音频）"""
    print(f"""
=== 测试音频准备 ===

请准备一段测试音频文件，建议：
1. 内容：美妆相关咨询，如"请推荐一款适合干皮的粉底液"
2. 格式：WAV, 16kHz采样率
3. 时长：3-10秒

保存路径: {output_path}

如果没有现成音频，可以使用以下方式生成：
- 使用 macOS 语音合成:
  say -o test_input.aiff "请推荐一款适合干皮的粉底液" &&
  ffmpeg -i test_input.aiff -ar 16000 test_input.wav

或者使用 Python 的 gTTS:
  pip install gtts && python -c "from gtts import gTTS; gTTS('请推荐一款适合干皮的粉底液', lang='zh').save('test_input.mp3')"
  ffmpeg -i test_input.mp3 -ar 16000 test_input.wav
""")


def main():
    """主函数"""
    print("=" * 60)
    print("MiniCPM-o 2.6 嘈杂环境语音测试")
    print("应用场景: 商场美妆集合店数字人")
    print("=" * 60)

    # 检查测试音频
    test_audio_path = "test_input.wav"
    if not os.path.exists(test_audio_path):
        create_test_audio(test_audio_path)
        print("\n请先准备测试音频文件，然后重新运行此脚本。")
        return

    # 加载模型
    model, tokenizer = load_model()

    # 测试语音识别
    print("\n" + "=" * 60)
    print("测试1: 不同噪声级别下的语音识别")
    print("=" * 60)
    recognition_results = test_speech_recognition(model, tokenizer, test_audio_path)

    # 测试对话
    print("\n" + "=" * 60)
    print("测试2: 嘈杂环境对话能力")
    print("=" * 60)
    dialogue_result = test_dialogue(model, tokenizer, test_audio_path)

    # 输出汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, desc, result in recognition_results:
        print(f"{desc}:")
        print(f"  {result[:100]}..." if len(str(result)) > 100 else f"  {result}")

    print("\n测试完成！输出文件保存在 outputs/ 目录")


if __name__ == "__main__":
    main()

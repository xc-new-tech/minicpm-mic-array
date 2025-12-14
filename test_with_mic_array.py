#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MiniCPM-o 2.6 + 麦克风阵列前端处理 完整测试

应用场景：商场美妆集合店数字人产品
测试目标：评估麦克风阵列 + 模型在嘈杂环境下的语音识别和对话能力

处理流程：
1. 麦克风阵列采集 -> 2. DOA声源定位 -> 3. 波束成形 -> 4. 降噪 -> 5. ASR/对话
"""

import os
import sys
import numpy as np
import torch
import librosa
import soundfile as sf
from transformers import AutoModel, AutoTokenizer

# 导入麦克风阵列处理模块
from mic_array import (
    MicArrayProcessor,
    create_linear_array,
    create_circular_array,
    simulate_mic_array_recording
)


def setup_flash_attn_mock():
    """绕过 flash_attn 依赖"""
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


def get_device():
    """获取可用设备"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model():
    """加载 MiniCPM-o 2.6 模型"""
    print("正在加载 MiniCPM-o 2.6 模型...")
    setup_flash_attn_mock()

    device = get_device()
    print(f"使用设备: {device}")

    model = AutoModel.from_pretrained(
        'openbmb/MiniCPM-o-2_6',
        trust_remote_code=True,
        attn_implementation='sdpa',
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

    model.init_tts()
    model.tts.float()

    print("模型加载完成!")
    return model, tokenizer


def test_with_mic_array(
    source_audio_path: str,
    model,
    tokenizer,
    output_dir: str = "outputs",
    noise_levels: list = None,
    use_separation: bool = False
):
    """
    使用麦克风阵列前端处理进行测试

    Args:
        source_audio_path: 源音频路径
        model: MiniCPM-o 模型
        tokenizer: tokenizer
        output_dir: 输出目录
        noise_levels: 噪声级别列表
        use_separation: 是否使用人声分离
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载源音频
    source_audio, sr = librosa.load(source_audio_path, sr=16000, mono=True)
    print(f"加载测试音频: {source_audio_path}")
    print(f"采样率: {sr} Hz, 长度: {len(source_audio) / sr:.2f} 秒")

    # 创建麦克风阵列（4麦克风线性阵列，间距5cm）
    mic_positions = create_linear_array(n_mics=4, spacing=0.05)
    processor = MicArrayProcessor(mic_positions, sample_rate=sr)

    if noise_levels is None:
        noise_levels = [
            ("moderate", 0.2, "中度噪声 (正常商场)"),
        ]

    results = []

    for name, noise_level, desc in noise_levels:
        print(f"\n{'=' * 60}")
        print(f"测试场景: {desc} (噪声级别: {noise_level})")
        print("=" * 60)

        # 模拟麦克风阵列录音
        print("\n1. 模拟商场环境录音...")
        multi_channel = simulate_mic_array_recording(
            source_audio,
            sr,
            mic_positions,
            source_direction=30,  # 声源在 30° 方向
            source_distance=1.0,
            noise_level=noise_level,
            room_dim=(6, 6, 3)
        )

        # 保存原始多通道录音的第一通道
        raw_path = os.path.join(output_dir, f"mic_array_{name}_raw.wav")
        sf.write(raw_path, multi_channel[0], sr)

        # === 处理1: 传统波束成形 ===
        print("\n2. 传统波束成形...")
        result_traditional = processor.process(
            multi_channel,
            auto_beamform=True,
            denoise=True,
            human_voice_only=False
        )
        traditional_path = os.path.join(output_dir, f"mic_array_{name}_traditional.wav")
        sf.write(traditional_path, result_traditional['audio'], sr)

        # === 处理2: 人声定位 + 人声分离 ===
        print("\n3. 人声定位 + 人声分离...")
        result_separated = processor.process(
            multi_channel,
            auto_beamform=True,
            denoise=True,
            human_voice_only=True,
            num_candidates=3,
            use_separation=True
        )
        separated_path = os.path.join(output_dir, f"mic_array_{name}_separated.wav")
        sf.write(separated_path, result_separated['audio'], sr)

        # === 处理3: 对分离后的音频再做人声分离（最终输出纯净人声）===
        print("\n4. 提取纯净人声...")
        final_vocals = processor.separate_voice(result_separated['audio'])
        final_vocals = final_vocals / (np.max(np.abs(final_vocals)) + 1e-8)
        vocals_path = os.path.join(output_dir, f"mic_array_{name}_vocals.wav")
        sf.write(vocals_path, final_vocals, sr)

        # ASR 测试
        print("\n5. MiniCPM-o 语音识别测试...")
        task_prompt = "请仔细听这段音频并转录内容。\n"

        # 测试原始录音
        print("   [1/4 原始录音]")
        raw_audio = multi_channel[0] / np.max(np.abs(multi_channel[0]) + 1e-8)
        try:
            res_raw = model.chat(
                msgs=[{'role': 'user', 'content': [task_prompt, raw_audio.astype(np.float32)]}],
                tokenizer=tokenizer,
                sampling=True,
                max_new_tokens=256,
                use_tts_template=False,
                generate_audio=False,
                temperature=0.3
            )
            print(f"       {res_raw}")
        except Exception as e:
            res_raw = f"ERROR: {e}"

        # 测试传统波束成形
        print("   [2/4 传统波束成形]")
        try:
            res_traditional = model.chat(
                msgs=[{'role': 'user', 'content': [task_prompt, result_traditional['audio']]}],
                tokenizer=tokenizer,
                sampling=True,
                max_new_tokens=256,
                use_tts_template=False,
                generate_audio=False,
                temperature=0.3
            )
            print(f"       {res_traditional}")
        except Exception as e:
            res_traditional = f"ERROR: {e}"

        # 测试人声定位+波束成形
        print("   [3/4 人声定位+波束成形]")
        try:
            res_separated = model.chat(
                msgs=[{'role': 'user', 'content': [task_prompt, result_separated['audio']]}],
                tokenizer=tokenizer,
                sampling=True,
                max_new_tokens=256,
                use_tts_template=False,
                generate_audio=False,
                temperature=0.3
            )
            print(f"       {res_separated}")
        except Exception as e:
            res_separated = f"ERROR: {e}"

        # 测试纯净人声
        print("   [4/4 纯净人声(Demucs)]")
        try:
            res_vocals = model.chat(
                msgs=[{'role': 'user', 'content': [task_prompt, final_vocals.astype(np.float32)]}],
                tokenizer=tokenizer,
                sampling=True,
                max_new_tokens=256,
                use_tts_template=False,
                generate_audio=False,
                temperature=0.3
            )
            print(f"       {res_vocals}")
        except Exception as e:
            res_vocals = f"ERROR: {e}"

        results.append({
            'scenario': desc,
            'noise_level': noise_level,
            'doa_traditional': result_traditional['doa']['azimuth'],
            'doa_separated': result_separated['doa']['azimuth'],
            'raw_result': res_raw,
            'traditional_result': res_traditional,
            'separated_result': res_separated,
            'vocals_result': res_vocals,
        })

    return results


def print_comparison_table(results: list, expected_text: str):
    """打印对比表格"""
    print("\n" + "=" * 100)
    print("测试结果对比")
    print("=" * 100)
    print(f"预期文本: {expected_text}")
    print("-" * 100)

    for r in results:
        print(f"\n场景: {r['scenario']}")
        print(f"  1. 原始录音:        {r['raw_result']}")
        print(f"  2. 传统波束成形:    {r['traditional_result']}")
        print(f"  3. 人声定位+波束:   {r['separated_result']}")
        print(f"  4. 纯净人声(Demucs): {r['vocals_result']}")

    print("\n" + "=" * 100)


def main():
    print("=" * 70)
    print("MiniCPM-o 2.6 + 麦克风阵列 + 人声分离 完整测试")
    print("应用场景: 商场美妆集合店数字人")
    print("=" * 70)

    # 检查测试音频
    test_audio_path = "test_input.wav"
    if not os.path.exists(test_audio_path):
        print(f"\n错误: 找不到测试音频 {test_audio_path}")
        print("请先运行 test_audio.py 创建测试音频")
        return

    # 加载模型
    model, tokenizer = load_model()

    # 运行测试 - 中度噪声
    print("\n开始测试...")
    noise_levels = [
        ("moderate", 0.2, "中度噪声 (正常商场)"),
    ]

    results = test_with_mic_array(
        test_audio_path,
        model,
        tokenizer,
        output_dir="outputs",
        noise_levels=noise_levels
    )

    # 打印对比表格
    expected_text = "你好，请推荐一款适合干皮的粉底液"
    print_comparison_table(results, expected_text)

    # 保存结果
    print("\n测试完成！")
    print("输出文件保存在 outputs/ 目录:")
    print("  - mic_array_*_raw.wav: 原始录音")
    print("  - mic_array_*_traditional.wav: 传统波束成形")
    print("  - mic_array_*_separated.wav: 人声定位+波束成形")
    print("  - mic_array_*_vocals.wav: 纯净人声(Demucs分离)")


if __name__ == "__main__":
    main()

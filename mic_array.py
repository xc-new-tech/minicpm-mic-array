#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
麦克风阵列前端处理模块

功能：
1. 声源定位 (DOA) - 识别响度最大的人声方向
2. 波束成形 (Beamforming) - 定向拾音
3. 语音活动检测 (VAD) - 检测人声
4. 降噪处理 - 抑制背景噪声

应用场景：商场美妆集合店数字人产品
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import pyroomacoustics as pra
import webrtcvad
import struct
import librosa
import torch


class MicArrayProcessor:
    """麦克风阵列处理器"""

    def __init__(
        self,
        mic_positions: np.ndarray,
        sample_rate: int = 16000,
        frame_size: int = 512,
        sound_speed: float = 343.0
    ):
        """
        初始化麦克风阵列处理器

        Args:
            mic_positions: 麦克风位置坐标 (n_mics, 3)，单位：米
                          例如：4麦克风线性阵列，间距5cm
                          np.array([[0, 0, 0], [0.05, 0, 0], [0.1, 0, 0], [0.15, 0, 0]])
            sample_rate: 采样率
            frame_size: 帧大小
            sound_speed: 声速 (m/s)
        """
        self.mic_positions = np.array(mic_positions)
        self.n_mics = len(mic_positions)
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.sound_speed = sound_speed

        # VAD 初始化
        self.vad = webrtcvad.Vad(3)  # 0-3, 3最激进

        # 波束成形器缓存
        self._beamformer = None
        self._last_doa = None

        # 人声分离模型 (延迟加载)
        self._separator = None
        self._separator_device = None

        print(f"麦克风阵列初始化完成:")
        print(f"  - 麦克风数量: {self.n_mics}")
        print(f"  - 采样率: {sample_rate} Hz")
        print(f"  - 帧大小: {frame_size}")

    def estimate_doa(self, multi_channel_audio: np.ndarray) -> dict:
        """
        估计声源方向 (Direction of Arrival)

        使用 SRP-PHAT 算法定位响度最大的声源

        Args:
            multi_channel_audio: 多通道音频 (n_samples, n_mics) 或 (n_mics, n_samples)

        Returns:
            dict: {
                'azimuth': 方位角 (度, 0-360),
                'elevation': 仰角 (度, -90 到 90),
                'energy': 该方向的能量值,
                'all_directions': 所有候选方向及能量
            }
        """
        # 确保音频格式正确 (n_mics, n_samples)
        if multi_channel_audio.shape[0] != self.n_mics:
            multi_channel_audio = multi_channel_audio.T

        # 使用 SRP-PHAT 算法
        nfft = 256
        freq_bins = np.arange(5, 60)  # 使用低频部分，更稳定

        doa = pra.doa.SRP(
            self.mic_positions.T,
            self.sample_rate,
            nfft,
            c=self.sound_speed,
            num_src=1,
            mode='far',
            azimuth=np.linspace(-180, 180, 360) * np.pi / 180,
            colatitude=np.linspace(0, np.pi, 90)
        )

        # 计算 STFT
        X = np.array([
            pra.transform.stft.analysis(
                multi_channel_audio[i],
                nfft,
                nfft // 2
            ).T
            for i in range(self.n_mics)
        ])

        # 定位
        doa.locate_sources(X, freq_bins=freq_bins)

        if len(doa.azimuth_recon) > 0:
            azimuth = np.degrees(doa.azimuth_recon[0])
            # 2D 问题没有 colatitude
            elevation = 0
            if hasattr(doa, 'colatitude_recon') and doa.colatitude_recon is not None:
                elevation = 90 - np.degrees(doa.colatitude_recon[0])

            # 归一化方位角到 0-360
            if azimuth < 0:
                azimuth += 360

            self._last_doa = azimuth

            return {
                'azimuth': azimuth,
                'elevation': elevation,
                'energy': float(np.max(doa.grid.values)),
                'confidence': 'high' if np.max(doa.grid.values) > 0.5 else 'low'
            }

        return {
            'azimuth': 0,
            'elevation': 0,
            'energy': 0,
            'confidence': 'none'
        }

    def beamform(
        self,
        multi_channel_audio: np.ndarray,
        target_direction: float = None,
        method: str = 'mvdr'
    ) -> np.ndarray:
        """
        波束成形 - 定向拾音

        Args:
            multi_channel_audio: 多通道音频 (n_samples, n_mics) 或 (n_mics, n_samples)
            target_direction: 目标方向 (度)，None 则自动使用 DOA 估计
            method: 波束成形方法 ('mvdr', 'das', 'gsc')

        Returns:
            np.ndarray: 波束成形后的单通道音频
        """
        # 确保音频格式正确 (n_mics, n_samples)
        if multi_channel_audio.shape[0] != self.n_mics:
            multi_channel_audio = multi_channel_audio.T

        # 如果没有指定方向，使用 DOA 估计
        if target_direction is None:
            doa_result = self.estimate_doa(multi_channel_audio)
            target_direction = doa_result['azimuth']
            print(f"  自动定位声源: {target_direction:.1f}° (置信度: {doa_result['confidence']})")

        # 转换为弧度
        azimuth_rad = np.radians(target_direction)

        # STFT 参数
        nfft = 512
        hop = nfft // 2

        # 计算 STFT
        X = np.array([
            pra.transform.stft.analysis(
                multi_channel_audio[i],
                nfft,
                hop
            ).T
            for i in range(self.n_mics)
        ])

        # 创建波束成形器 - 使用 Delay-and-Sum（更稳定）
        # 计算目标位置
        target_distance = 1.0  # 假设目标在 1 米处
        target_pos = np.array([
            [np.cos(azimuth_rad) * target_distance],
            [np.sin(azimuth_rad) * target_distance],
            [0]
        ])

        # 使用简单的 delay-and-sum 波束成形
        # 计算每个麦克风到目标的延迟
        delays = np.zeros(self.n_mics)
        for i in range(self.n_mics):
            mic_pos = self.mic_positions[i]
            # 计算波程差（相对于阵列中心）
            path_diff = np.cos(azimuth_rad) * mic_pos[0] + np.sin(azimuth_rad) * mic_pos[1]
            delays[i] = path_diff / self.sound_speed

        # 延迟补偿后求和
        max_delay_samples = int(np.max(np.abs(delays)) * self.sample_rate) + 10
        output_len = multi_channel_audio.shape[1]
        output = np.zeros(output_len)

        for i in range(self.n_mics):
            delay_samples = int(delays[i] * self.sample_rate)
            if delay_samples >= 0:
                if delay_samples < output_len:
                    output[delay_samples:] += multi_channel_audio[i, :output_len - delay_samples]
            else:
                delay_samples = -delay_samples
                if delay_samples < output_len:
                    output[:output_len - delay_samples] += multi_channel_audio[i, delay_samples:]

        output = output / self.n_mics
        return output

    def detect_voice_activity(
        self,
        audio: np.ndarray,
        frame_duration_ms: int = 30
    ) -> list:
        """
        语音活动检测 (VAD)

        Args:
            audio: 单通道音频
            frame_duration_ms: 帧长度 (10, 20, 或 30 ms)

        Returns:
            list: [(start_sample, end_sample, is_speech), ...]
        """
        # 确保是 16-bit PCM
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)

        frame_size = int(self.sample_rate * frame_duration_ms / 1000)
        frames = []

        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i + frame_size]
            frame_bytes = struct.pack(f'{len(frame)}h', *frame)

            try:
                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
            except:
                is_speech = False

            frames.append((i, i + frame_size, is_speech))

        # 合并连续的语音段
        merged = []
        if frames:
            start, end, current_speech = frames[0]
            for i in range(1, len(frames)):
                if frames[i][2] == current_speech:
                    end = frames[i][1]
                else:
                    merged.append((start, end, current_speech))
                    start, end, current_speech = frames[i]
            merged.append((start, end, current_speech))

        return merged

    def _load_separator(self):
        """延迟加载人声分离模型 (Demucs)"""
        if self._separator is None:
            print("    加载人声分离模型 (Demucs htdemucs)...")
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            # 选择设备
            if torch.backends.mps.is_available():
                self._separator_device = torch.device("mps")
            elif torch.cuda.is_available():
                self._separator_device = torch.device("cuda")
            else:
                self._separator_device = torch.device("cpu")

            # 加载模型 (htdemucs 是最新最好的模型)
            self._separator = get_model('htdemucs')
            self._separator.to(self._separator_device)
            self._separator.eval()
            print(f"    模型加载完成 (设备: {self._separator_device})")

        return self._separator

    def separate_voice(
        self,
        audio: np.ndarray,
        return_all: bool = False
    ) -> np.ndarray:
        """
        人声分离 - 从混合音频中提取纯净人声

        使用 Demucs (htdemucs) 模型分离人声

        Args:
            audio: 输入音频 (单通道, float)
            return_all: 是否返回所有分离的音轨

        Returns:
            np.ndarray: 分离后的人声
            如果 return_all=True, 返回 dict: {'vocals', 'drums', 'bass', 'other'}
        """
        from demucs.apply import apply_model

        model = self._load_separator()

        # 确保音频格式正确
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32767.0

        # 归一化
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        # Demucs 需要 44100Hz，如果不是则重采样
        if self.sample_rate != 44100:
            audio_44k = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=44100)
        else:
            audio_44k = audio

        # 转换为 torch tensor: (batch, channels, samples)
        # Demucs 需要立体声，单声道复制为双声道
        audio_tensor = torch.tensor(audio_44k, dtype=torch.float32)
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        audio_tensor = audio_tensor.repeat(1, 2, 1)  # (1, 2, samples) 立体声
        audio_tensor = audio_tensor.to(self._separator_device)

        # 分离
        with torch.no_grad():
            sources = apply_model(model, audio_tensor, device=self._separator_device)
            # sources shape: (batch, num_sources, channels, samples)
            # num_sources: drums, bass, other, vocals (顺序取决于模型)

        # 获取人声 (vocals 是最后一个)
        source_names = model.sources  # ['drums', 'bass', 'other', 'vocals']
        vocals_idx = source_names.index('vocals')

        vocals = sources[0, vocals_idx].cpu().numpy()
        # 转为单声道
        vocals_mono = np.mean(vocals, axis=0)

        # 重采样回原始采样率
        if self.sample_rate != 44100:
            vocals_mono = librosa.resample(vocals_mono, orig_sr=44100, target_sr=self.sample_rate)

        # 匹配原始长度
        if len(vocals_mono) > len(audio):
            vocals_mono = vocals_mono[:len(audio)]
        elif len(vocals_mono) < len(audio):
            vocals_mono = np.pad(vocals_mono, (0, len(audio) - len(vocals_mono)))

        if return_all:
            result = {}
            for i, name in enumerate(source_names):
                src = sources[0, i].cpu().numpy()
                src_mono = np.mean(src, axis=0)
                if self.sample_rate != 44100:
                    src_mono = librosa.resample(src_mono, orig_sr=44100, target_sr=self.sample_rate)
                if len(src_mono) > len(audio):
                    src_mono = src_mono[:len(audio)]
                elif len(src_mono) < len(audio):
                    src_mono = np.pad(src_mono, (0, len(audio) - len(src_mono)))
                result[name] = src_mono.astype(np.float32)
            return result

        return vocals_mono.astype(np.float32)

    def is_human_voice(
        self,
        audio: np.ndarray,
        threshold: float = 0.5,
        use_separation: bool = False
    ) -> dict:
        """
        判断音频是否为人声

        使用多种特征综合判断：
        1. 基频检测 (F0) - 人声基频范围 85-255Hz
        2. VAD 语音活动比例
        3. 频谱能量分布 - 人声主要在 300-3400Hz
        4. (可选) 人声分离后的能量比例

        Args:
            audio: 单通道音频 (float, -1到1)
            threshold: 人声判断阈值 (0-1)
            use_separation: 是否使用人声分离来提高检测准确性

        Returns:
            dict: {
                'is_human': bool,
                'confidence': float (0-1),
                'f0_ratio': 基频有效帧比例,
                'vad_ratio': 语音活动比例,
                'spectral_score': 频谱人声特征得分,
                'separation_ratio': 人声分离能量比 (仅当 use_separation=True)
            }
        """
        # 确保音频是 float 格式
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32767.0

        # 归一化
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        # 如果使用人声分离，先分离出人声再检测
        analysis_audio = audio
        separation_ratio = None
        if use_separation:
            vocals = self.separate_voice(audio)
            # 计算人声能量占比
            original_energy = np.sum(audio ** 2)
            vocals_energy = np.sum(vocals ** 2)
            separation_ratio = vocals_energy / (original_energy + 1e-8)
            # 使用分离后的人声进行后续分析
            analysis_audio = vocals

        # 1. 基频检测 (F0) - 使用分析音频
        f0, voiced_flag, _ = librosa.pyin(
            analysis_audio,
            fmin=80,   # 最低基频
            fmax=300,  # 最高基频
            sr=self.sample_rate
        )
        # 有效基频帧比例 (人声基频在 85-255Hz)
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            human_f0_mask = (valid_f0 >= 85) & (valid_f0 <= 255)
            f0_ratio = np.sum(human_f0_mask) / len(f0)
        else:
            f0_ratio = 0.0

        # 2. VAD 比例 - 使用分析音频
        vad_result = self.detect_voice_activity(analysis_audio)
        total_samples = len(analysis_audio)
        speech_samples = sum(e - s for s, e, is_speech in vad_result if is_speech)
        vad_ratio = speech_samples / total_samples if total_samples > 0 else 0

        # 3. 频谱能量分布 - 使用分析音频
        n_fft = 2048
        spec = np.abs(librosa.stft(analysis_audio, n_fft=n_fft))
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)

        # 人声频段能量 (300-3400Hz)
        voice_band = (freqs >= 300) & (freqs <= 3400)
        total_energy = np.sum(spec ** 2)
        voice_energy = np.sum(spec[voice_band, :] ** 2)
        spectral_score = voice_energy / (total_energy + 1e-8)

        # 综合评分
        if use_separation and separation_ratio is not None:
            # 加入人声分离能量比作为额外特征
            # 如果分离出的人声能量占比高，更可能是人声
            confidence = (
                f0_ratio * 0.3 +
                vad_ratio * 0.2 +
                spectral_score * 0.2 +
                min(separation_ratio, 1.0) * 0.3  # 人声能量占比
            )
        else:
            confidence = (f0_ratio * 0.4 + vad_ratio * 0.3 + spectral_score * 0.3)

        is_human = confidence >= threshold

        result = {
            'is_human': is_human,
            'confidence': float(confidence),
            'f0_ratio': float(f0_ratio),
            'vad_ratio': float(vad_ratio),
            'spectral_score': float(spectral_score)
        }

        if separation_ratio is not None:
            result['separation_ratio'] = float(separation_ratio)

        return result

    def estimate_doa_multi(
        self,
        multi_channel_audio: np.ndarray,
        num_sources: int = 3
    ) -> list:
        """
        估计多个声源方向

        Args:
            multi_channel_audio: 多通道音频 (n_mics, n_samples)
            num_sources: 候选声源数量

        Returns:
            list: [{'azimuth': float, 'energy': float}, ...]
        """
        # 确保音频格式正确
        if multi_channel_audio.shape[0] != self.n_mics:
            multi_channel_audio = multi_channel_audio.T

        nfft = 256
        freq_bins = np.arange(5, 60)

        doa = pra.doa.SRP(
            self.mic_positions.T,
            self.sample_rate,
            nfft,
            c=self.sound_speed,
            num_src=num_sources,
            mode='far',
            azimuth=np.linspace(-180, 180, 360) * np.pi / 180,
            colatitude=np.linspace(0, np.pi, 90)
        )

        # 计算 STFT
        X = np.array([
            pra.transform.stft.analysis(
                multi_channel_audio[i],
                nfft,
                nfft // 2
            ).T
            for i in range(self.n_mics)
        ])

        doa.locate_sources(X, freq_bins=freq_bins)

        # 获取能量分布
        grid_values = doa.grid.values
        azimuths_rad = doa.grid.azimuth

        # 找到峰值（局部最大值）
        sources = []
        for i in range(len(doa.azimuth_recon)):
            azimuth = np.degrees(doa.azimuth_recon[i])
            if azimuth < 0:
                azimuth += 360
            energy = float(grid_values[i]) if i < len(grid_values) else 0
            sources.append({'azimuth': azimuth, 'energy': energy})

        # 按能量排序
        sources.sort(key=lambda x: x['energy'], reverse=True)

        return sources[:num_sources]

    def find_human_voice_direction(
        self,
        multi_channel_audio: np.ndarray,
        num_candidates: int = 3,
        use_separation: bool = False
    ) -> dict:
        """
        找到人声方向

        流程：
        1. 定位多个候选声源
        2. 对每个方向进行波束成形
        3. (可选) 人声分离
        4. 检测是否为人声
        5. 返回人声置信度最高的方向

        Args:
            multi_channel_audio: 多通道音频
            num_candidates: 候选声源数量
            use_separation: 是否使用人声分离来提高检测准确性

        Returns:
            dict: {
                'azimuth': 人声方向,
                'confidence': 置信度,
                'is_human': 是否检测到人声,
                'all_candidates': 所有候选及其人声评分
            }
        """
        sep_hint = " (含人声分离)" if use_separation else ""
        print(f"  正在定位人声方向{sep_hint}...")

        # 确保格式正确
        if multi_channel_audio.shape[0] != self.n_mics:
            multi_channel_audio = multi_channel_audio.T

        # 1. 定位多个候选声源
        candidates = self.estimate_doa_multi(multi_channel_audio, num_candidates)
        candidate_angles = [f"{c['azimuth']:.1f}°" for c in candidates]
        print(f"    候选声源: {candidate_angles}")

        # 2. 对每个方向检测人声
        results = []
        for i, candidate in enumerate(candidates):
            direction = candidate['azimuth']

            # 波束成形到该方向
            beam_audio = self.beamform(multi_channel_audio, direction)
            beam_audio = beam_audio / (np.max(np.abs(beam_audio)) + 1e-8)

            # 检测是否为人声 (可选人声分离)
            voice_result = self.is_human_voice(beam_audio, use_separation=use_separation)

            result_entry = {
                'azimuth': direction,
                'energy': candidate['energy'],
                'is_human': voice_result['is_human'],
                'human_confidence': voice_result['confidence'],
                'f0_ratio': voice_result['f0_ratio'],
                'vad_ratio': voice_result['vad_ratio']
            }
            if 'separation_ratio' in voice_result:
                result_entry['separation_ratio'] = voice_result['separation_ratio']

            results.append(result_entry)

            status = "✓ 人声" if voice_result['is_human'] else "✗ 非人声"
            extra_info = ""
            if 'separation_ratio' in voice_result:
                extra_info = f", 分离比:{voice_result['separation_ratio']:.2f}"
            print(f"    {direction:.1f}°: {status} (置信度: {voice_result['confidence']:.2f}{extra_info})")

        # 3. 选择人声置信度最高的方向
        # 优先选择确认为人声的
        human_results = [r for r in results if r['is_human']]
        if human_results:
            best = max(human_results, key=lambda x: x['human_confidence'])
        else:
            # 如果没有确认的人声，选择置信度最高的
            best = max(results, key=lambda x: x['human_confidence'])

        print(f"    选定方向: {best['azimuth']:.1f}° (人声置信度: {best['human_confidence']:.2f})")

        return {
            'azimuth': best['azimuth'],
            'confidence': best['human_confidence'],
            'is_human': best['is_human'],
            'all_candidates': results
        }

    def noise_reduce(
        self,
        audio: np.ndarray,
        noise_profile: np.ndarray = None
    ) -> np.ndarray:
        """
        降噪处理

        使用谱减法进行降噪

        Args:
            audio: 输入音频
            noise_profile: 噪声样本（用于估计噪声频谱），None 则使用前 0.5 秒

        Returns:
            np.ndarray: 降噪后的音频
        """
        # 使用前 0.5 秒作为噪声估计
        if noise_profile is None:
            noise_samples = int(0.5 * self.sample_rate)
            noise_profile = audio[:noise_samples]

        # STFT 参数
        nfft = 512
        hop = nfft // 2

        # 计算噪声频谱
        noise_stft = pra.transform.stft.analysis(noise_profile, nfft, hop)
        noise_power = np.mean(np.abs(noise_stft) ** 2, axis=0)

        # 计算信号 STFT
        signal_stft = pra.transform.stft.analysis(audio, nfft, hop)

        # 谱减法
        signal_power = np.abs(signal_stft) ** 2
        clean_power = np.maximum(signal_power - noise_power * 2, 0.01 * signal_power)
        clean_magnitude = np.sqrt(clean_power)

        # 保持相位
        phase = np.angle(signal_stft)
        clean_stft = clean_magnitude * np.exp(1j * phase)

        # 逆 STFT
        clean_audio = pra.transform.stft.synthesis(clean_stft, nfft, hop)

        # 匹配原始长度
        if len(clean_audio) > len(audio):
            clean_audio = clean_audio[:len(audio)]
        elif len(clean_audio) < len(audio):
            clean_audio = np.pad(clean_audio, (0, len(audio) - len(clean_audio)))

        return clean_audio

    def process(
        self,
        multi_channel_audio: np.ndarray,
        auto_beamform: bool = True,
        denoise: bool = True,
        human_voice_only: bool = False,
        num_candidates: int = 3,
        use_separation: bool = False
    ) -> dict:
        """
        完整的前端处理流程

        1. 声源定位 (DOA) - 可选仅定位人声
        2. 波束成形 (定向拾音)
        3. 语音活动检测 (VAD)
        4. 降噪

        Args:
            multi_channel_audio: 多通道音频 (n_samples, n_mics) 或 (n_mics, n_samples)
            auto_beamform: 是否自动波束成形
            denoise: 是否降噪
            human_voice_only: 是否只聚焦人声方向 (过滤音乐、噪声等)
            num_candidates: 人声定位时的候选声源数量
            use_separation: 是否使用人声分离来提高人声检测准确性

        Returns:
            dict: {
                'audio': 处理后的音频,
                'doa': 声源方向信息,
                'vad': 语音活动段,
                'sample_rate': 采样率,
                'human_voice_detection': 人声检测结果 (仅当 human_voice_only=True)
            }
        """
        print("开始麦克风阵列前端处理...")

        # 确保格式正确
        if multi_channel_audio.shape[0] != self.n_mics:
            multi_channel_audio = multi_channel_audio.T

        # 1. 声源定位
        human_voice_result = None
        if human_voice_only:
            # 使用人声定位
            sep_hint = " + 人声分离" if use_separation else ""
            print(f"  1. 人声定位{sep_hint} (过滤非人声)...")
            human_voice_result = self.find_human_voice_direction(
                multi_channel_audio, num_candidates, use_separation=use_separation
            )
            doa_result = {
                'azimuth': human_voice_result['azimuth'],
                'elevation': 0,
                'energy': human_voice_result['confidence'],
                'confidence': 'high' if human_voice_result['is_human'] else 'low'
            }
            if not human_voice_result['is_human']:
                print("     ⚠ 未检测到明确人声，使用最可能方向")
        else:
            # 传统定位（最强声源）
            print("  1. 声源定位 (DOA)...")
            doa_result = self.estimate_doa(multi_channel_audio)
            print(f"     方位角: {doa_result['azimuth']:.1f}°, 置信度: {doa_result['confidence']}")

        # 2. 波束成形
        if auto_beamform:
            print("  2. 波束成形 (定向拾音)...")
            audio = self.beamform(multi_channel_audio, doa_result['azimuth'])
        else:
            # 简单混合
            audio = np.mean(multi_channel_audio, axis=0)

        # 归一化
        audio = audio / np.max(np.abs(audio) + 1e-8)

        # 3. 降噪
        if denoise:
            print("  3. 降噪处理...")
            audio = self.noise_reduce(audio)
            audio = audio / np.max(np.abs(audio) + 1e-8)

        # 4. VAD
        print("  4. 语音活动检测 (VAD)...")
        vad_result = self.detect_voice_activity(audio)
        speech_segments = [(s, e) for s, e, is_speech in vad_result if is_speech]
        print(f"     检测到 {len(speech_segments)} 个语音段")

        print("前端处理完成!")

        result = {
            'audio': audio.astype(np.float32),
            'doa': doa_result,
            'vad': vad_result,
            'speech_segments': speech_segments,
            'sample_rate': self.sample_rate
        }

        if human_voice_result is not None:
            result['human_voice_detection'] = human_voice_result

        return result


def create_linear_array(n_mics: int = 4, spacing: float = 0.05) -> np.ndarray:
    """
    创建线性麦克风阵列

    Args:
        n_mics: 麦克风数量
        spacing: 麦克风间距 (米)

    Returns:
        np.ndarray: 麦克风位置 (n_mics, 3)
    """
    positions = np.zeros((n_mics, 3))
    for i in range(n_mics):
        positions[i, 0] = i * spacing
    return positions


def create_circular_array(n_mics: int = 6, radius: float = 0.05) -> np.ndarray:
    """
    创建圆形麦克风阵列

    Args:
        n_mics: 麦克风数量
        radius: 阵列半径 (米)

    Returns:
        np.ndarray: 麦克风位置 (n_mics, 3)
    """
    angles = np.linspace(0, 2 * np.pi, n_mics, endpoint=False)
    positions = np.zeros((n_mics, 3))
    positions[:, 0] = radius * np.cos(angles)
    positions[:, 1] = radius * np.sin(angles)
    return positions


def simulate_mic_array_recording(
    source_audio: np.ndarray,
    sample_rate: int,
    mic_positions: np.ndarray,
    source_direction: float = 0,  # 度
    source_distance: float = 1.0,  # 米
    noise_level: float = 0.1,
    room_dim: tuple = (5, 5, 3)  # 房间尺寸 (米)
) -> np.ndarray:
    """
    模拟麦克风阵列录音

    Args:
        source_audio: 源音频
        sample_rate: 采样率
        mic_positions: 麦克风位置
        source_direction: 声源方向 (度)
        source_distance: 声源距离 (米)
        noise_level: 噪声级别
        room_dim: 房间尺寸

    Returns:
        np.ndarray: 多通道录音 (n_mics, n_samples)
    """
    n_mics = len(mic_positions)

    # 计算声源位置
    azimuth_rad = np.radians(source_direction)
    source_pos = np.array([
        room_dim[0] / 2 + source_distance * np.cos(azimuth_rad),
        room_dim[1] / 2 + source_distance * np.sin(azimuth_rad),
        1.5  # 人声高度
    ])

    # 麦克风位置（放在房间中心）
    mic_center = np.array([room_dim[0] / 2, room_dim[1] / 2, 1.5])
    mic_locs = mic_positions + mic_center

    # 创建房间
    room = pra.ShoeBox(
        room_dim,
        fs=sample_rate,
        materials=pra.Material(0.3),  # 吸声系数
        max_order=3  # 反射阶数
    )

    # 添加麦克风阵列
    room.add_microphone_array(mic_locs.T)

    # 添加声源
    room.add_source(source_pos, signal=source_audio)

    # 模拟
    room.simulate()

    # 获取录音
    recording = room.mic_array.signals

    # 添加噪声
    noise = np.random.randn(*recording.shape) * noise_level
    recording = recording + noise

    return recording


# 测试代码
if __name__ == "__main__":
    import soundfile as sf
    import os

    print("=" * 60)
    print("麦克风阵列前端处理测试 (含人声定位)")
    print("=" * 60)

    os.makedirs("outputs", exist_ok=True)

    # 创建 4 麦克风线性阵列（间距 5cm）
    mic_positions = create_linear_array(n_mics=4, spacing=0.05)
    print(f"\n麦克风阵列配置: 4麦克风线性阵列, 间距5cm")

    # 加载测试音频
    audio_path = "test_input.wav"
    source_audio, sr = sf.read(audio_path)
    print(f"加载测试音频: {audio_path}")
    print(f"采样率: {sr} Hz, 长度: {len(source_audio) / sr:.2f} 秒")

    # 模拟商场环境：人声在 30°，背景音乐在 120°
    print("\n" + "-" * 60)
    print("模拟商场环境: 人声(30°) + 背景噪声")
    print("-" * 60)

    multi_channel = simulate_mic_array_recording(
        source_audio,
        sr,
        mic_positions,
        source_direction=30,
        source_distance=1.0,
        noise_level=0.2
    )

    # 创建处理器
    processor = MicArrayProcessor(mic_positions, sample_rate=sr)

    # 测试1: 传统定位（最强声源）
    print("\n【测试1: 传统定位 (最强声源)】")
    result1 = processor.process(
        multi_channel,
        auto_beamform=True,
        denoise=True,
        human_voice_only=False
    )
    sf.write("outputs/traditional_beamform.wav", result1['audio'], sr)

    # 测试2: 人声定位（无人声分离）
    print("\n【测试2: 人声定位 (无人声分离)】")
    result2 = processor.process(
        multi_channel,
        auto_beamform=True,
        denoise=True,
        human_voice_only=True,
        num_candidates=3,
        use_separation=False
    )
    sf.write("outputs/human_voice_beamform.wav", result2['audio'], sr)

    # 测试3: 人声定位 + 人声分离
    print("\n【测试3: 人声定位 + 人声分离 (Demucs)】")
    result3 = processor.process(
        multi_channel,
        auto_beamform=True,
        denoise=True,
        human_voice_only=True,
        num_candidates=3,
        use_separation=True
    )
    sf.write("outputs/human_voice_separated_beamform.wav", result3['audio'], sr)

    # 结果对比
    print("\n" + "=" * 60)
    print("结果对比")
    print("=" * 60)
    print(f"真实人声方向: 30°")
    print(f"传统定位结果: {result1['doa']['azimuth']:.1f}°")
    print(f"人声定位结果: {result2['doa']['azimuth']:.1f}°")
    print(f"人声定位+分离: {result3['doa']['azimuth']:.1f}°")

    # 显示人声分离版本的详情
    if 'human_voice_detection' in result3:
        hvd = result3['human_voice_detection']
        print(f"\n人声检测详情 (含分离):")
        print(f"  是否人声: {'是' if hvd['is_human'] else '否'}")
        print(f"  置信度: {hvd['confidence']:.2f}")
        print(f"  候选方向:")
        for c in hvd['all_candidates']:
            status = "✓" if c['is_human'] else "✗"
            sep_info = f", 分离比:{c['separation_ratio']:.2f}" if 'separation_ratio' in c else ""
            print(f"    {status} {c['azimuth']:.1f}° - 置信度:{c['human_confidence']:.2f}, "
                  f"F0:{c['f0_ratio']:.2f}, VAD:{c['vad_ratio']:.2f}{sep_info}")

    print("\n输出文件:")
    print("  - outputs/traditional_beamform.wav (传统定位)")
    print("  - outputs/human_voice_beamform.wav (人声定位)")
    print("  - outputs/human_voice_separated_beamform.wav (人声定位+分离)")
    print("=" * 60)

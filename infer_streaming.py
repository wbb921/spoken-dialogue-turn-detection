#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对已训练的 DualStreamMimiModel 进行流式推理的脚本。
该脚本模拟实时音频流，以80ms为块进行处理，并维护一个20秒的滑动上下文窗口。

author: 陆海天 (adapted by Gemini)
"""

import os
import argparse
import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor
from model.DualStreamMimiModel import DualStreamMimiModel # 导入您的模型类
from tqdm import tqdm

# 状态索引到人类可读名称的映射。
STATE_MAP = {
    0: "Speak",
    1: "Listen",
    2: "Gap",
    3: "Pause",
    4: "Overlap",
}

class StreamProcessor:
    """
    一个有状态的处理器，用于管理音频缓冲区并执行流式推理。
    """
    def __init__(self, model, feature_extractor, device, args):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        self.args = args
        
        # --- 配置 ---
        self.target_sr = feature_extractor.sampling_rate
        self.context_s = 10  # 上下文窗口长度（秒）
        self.chunk_s = 0.080 # 每个处理块的长度（秒），即80ms
        
        # 将时间转换为采样点数
        self.context_samples = int(self.context_s * self.target_sr)
        self.chunk_samples = int(self.chunk_s * self.target_sr)
        
        # --- 状态缓冲区 ---
        self.user_buffer = torch.zeros(self.context_samples, dtype=torch.float32)
        self.system_buffer = torch.zeros(self.context_samples, dtype=torch.float32)
        
        # --- 状态跟踪 ---
        self.user_stream_state = {"current": -1, "start_time": 0.0}
        self.system_stream_state = {"current": -1, "start_time": 0.0}
        self.current_time = 0.0

    def _run_model_on_context(self):
        """在当前缓冲区上运行模型并返回最后一帧的预测。"""
        # 准备模型输入
        user_feats = torch.tensor(self.feature_extractor(self.user_buffer, sampling_rate=self.target_sr)["input_values"][0]).T
        system_feats = torch.tensor(self.feature_extractor(self.system_buffer, sampling_rate=self.target_sr)["input_values"][0]).T
        
        user_input_values = user_feats.unsqueeze(0).transpose(1, 2).to(self.device)
        system_input_values = system_feats.unsqueeze(0).transpose(1, 2).to(self.device)
        
        user_padding_mask = torch.ones(1, user_feats.size(0), dtype=torch.bool).to(self.device)
        system_padding_mask = torch.ones(1, system_feats.size(0), dtype=torch.bool).to(self.device)

        with torch.no_grad():
            out = self.model(
                user_input_values=user_input_values, user_padding_mask=user_padding_mask,
                system_input_values=system_input_values, system_padding_mask=system_padding_mask,
                look_ahead=self.args.look_ahead
            )
        
        logits_u, logits_s = out["user_output"], out["system_output"]
        
        # 我们只关心对应于最新音频块的最后一帧的预测
        pred_u = torch.argmax(logits_u.squeeze(0), dim=-1)[-1].item()
        pred_s = torch.argmax(logits_s.squeeze(0), dim=-1)[-1].item()
        
        return pred_u, pred_s

    def _update_and_print_transitions(self, stream_name, stream_state, prediction):
        """检查状态变化并打印已完成的片段。"""
        if stream_state["current"] == -1: # 初始化
            stream_state["current"] = prediction
        
        if prediction != stream_state["current"]:
            end_time = self.current_time
            start_time = stream_state["start_time"]
            duration = end_time - start_time
            state_id = stream_state["current"]
            state_name = STATE_MAP.get(state_id, f"Unknown State {state_id}")

            print(f"  [{stream_name}] {(start_time - 0.08):>7.2f}s -> {(end_time - 0.08):>7.2f}s ({duration:>6.2f}s) | 状态: {state_name}")
            
            stream_state["current"] = prediction
            stream_state["start_time"] = end_time

    def process(self, user_chunk, system_chunk):
        """处理一个新的80ms音频块。"""
        # 更新缓冲区：向右滚动并添加新数据
        self.user_buffer = torch.roll(self.user_buffer, shifts=-self.chunk_samples, dims=0)
        self.user_buffer[-self.chunk_samples:] = user_chunk
        
        self.system_buffer = torch.roll(self.system_buffer, shifts=-self.chunk_samples, dims=0)
        self.system_buffer[-self.chunk_samples:] = system_chunk
        
        # 运行推理
        pred_u, pred_s = self._run_model_on_context()
        
        # 更新时间戳
        self.current_time += self.chunk_s
        
        # 检查并报告状态转移
        self._update_and_print_transitions("用户流", self.user_stream_state, pred_u)
        self._update_and_print_transitions("系统流", self.system_stream_state, pred_s)
        
    def flush(self):
        """处理结束后，打印最后一个正在进行的片段。"""
        self._update_and_print_transitions("用户流", self.user_stream_state, -1) # 使用一个无效状态来强制结束
        self._update_and_print_transitions("系统流", self.system_stream_state, -1)


def run_streaming_inference(model, audio_path: str, feature_extractor, device, args):
    """
    加载音频文件并以流式方式进行处理。
    """
    print(f"--- 正在处理音频文件: {audio_path} ---")
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"[错误] 无法加载音频文件: {e}")
        return

    if waveform.shape[0] != 2:
        print(f"[错误] 需要立体声音频，但文件有 {waveform.shape[0]} 个声道。")
        return
        
    user_audio_full, system_audio_full = waveform[0], waveform[1]
    
    target_sr = feature_extractor.sampling_rate
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        user_audio_full = resampler(user_audio_full)
        system_audio_full = resampler(system_audio_full)

    # 实例化流处理器
    stream_processor = StreamProcessor(model, feature_extractor, device, args)
    chunk_samples = stream_processor.chunk_samples
    total_samples = user_audio_full.shape[0]

    print("\n## 实时状态转移矩阵 ##")
    pbar = tqdm(total=total_samples, unit="samples", desc="流式推理进度")

    # 模拟音频流，以80ms为块进行处理
    for start_sample in range(0, total_samples, chunk_samples):
        end_sample = start_sample + chunk_samples
        if end_sample > total_samples:
            break # 忽略最后一个不足80ms的片段

        user_chunk = user_audio_full[start_sample:end_sample]
        system_chunk = system_audio_full[start_sample:end_sample]
        
        stream_processor.process(user_chunk, system_chunk)
        pbar.update(chunk_samples)
    
    pbar.close()
    stream_processor.flush() # 结束并打印最后一个片段


def parse_args():
    ap = argparse.ArgumentParser(description="使用80ms块和20s上下文窗口对DualStreamMimi模型进行流式推理。")
    ap.add_argument("--audio_path", required=True, type=str, help="输入的立体声.wav文件路径。")
    ap.add_argument("--checkpoint_path", required=True, type=str, help="训练好的模型检查点 (.pt) 文件路径。")
    ap.add_argument("--model_name", default="kyutai/mimi", help="用于特征提取器的Hugging Face基础模型名称。")
    ap.add_argument("--num_classes", type=int, default=5, help="模型的输出类别数量。")
    ap.add_argument("--look_ahead", type=int, default=0, help="模型训练时使用的look_ahead值。")
    return ap.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # --- 加载模型 ---
    print(f"正在从以下路径加载模型检查点: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        print(f"错误: 找不到检查点路径: {args.checkpoint_path}")
        return
    model = DualStreamMimiModel(num_classes=args.num_classes).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state', checkpoint)
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("模型加载成功。")

    # --- 加载特征提取器 ---
    print(f"正在加载特征提取器: {args.model_name}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    
    # --- 运行流式推理 ---
    run_streaming_inference(model, args.audio_path, feature_extractor, device, args)

if __name__ == "__main__":
    main()
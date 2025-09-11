#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor
from model.DualStreamMimiModel import DualStreamMimiModel # 导入您的模型类
from tqdm import tqdm

STATE_MAP = {
    0: "Speak",
    1: "Listen",
    2: "Gap",
    3: "Pause",
    4: "Overlap",
}

def generate_state_transitions(prediction_sequence: np.ndarray, frame_duration_s: float) -> list:
    """
    将帧级别的状态预测序列转换为状态转移列表。
    """
    if prediction_sequence.size == 0:
        return []

    transitions = []
    current_state = prediction_sequence[0]
    start_time = 0.0

    for i, state in enumerate(prediction_sequence):
        if state != current_state:
            end_time = i * frame_duration_s
            transitions.append({
                "start": f"{start_time:.2f}s",
                "end": f"{end_time:.2f}s",
                "duration": f"{end_time - start_time:.2f}s",
                "state_id": int(current_state),
                "state_name": STATE_MAP.get(current_state, f"Unknown State {current_state}")
            })
            current_state = state
            start_time = end_time

    end_time = len(prediction_sequence) * frame_duration_s
    transitions.append({
        "start": f"{start_time:.2f}s",
        "end": f"{end_time:.2f}s",
        "duration": f"{end_time - start_time:.2f}s",
        "state_id": int(current_state),
        "state_name": STATE_MAP.get(current_state, f"Unknown State {current_state}")
    })

    return transitions

def run_inference(model, audio_path: str, feature_extractor, device, args):
    """
    加载单个音频文件，以20秒为窗口进行推理，并打印最终的状态转移矩阵。
    """
    model.eval()
    print(f"--- 正在处理音频文件: {audio_path} ---")

    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"[错误] 无法加载音频文件: {e}")
        return

    if waveform.shape[0] != 2:
        print(f"[错误] 需要一个包含2个声道的立体声音频文件，但当前文件有 {waveform.shape[0]} 个声道。")
        return
        
    user_audio_full, system_audio_full = waveform[0], waveform[1]
    
    target_sr = feature_extractor.sampling_rate
    if sr != target_sr:
        print(f"正在将音频从 {sr} Hz 重采样至 {target_sr} Hz...")
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        user_audio_full = resampler(user_audio_full)
        system_audio_full = resampler(system_audio_full)

    # --- 窗口化处理设置 ---
    window_len_s = 20  # 窗口长度为20秒
    segment_samples = window_len_s * target_sr
    
    all_preds_u = []
    all_preds_s = []

    total_samples = user_audio_full.shape[0]
    
    print(f"开始以 {window_len_s}s 为窗口进行推理...")
    # 使用 tqdm 创建进度条
    pbar = tqdm(total=total_samples, unit="samples", desc="推理进度")

    # --- 循环遍历每个20秒的窗口 ---
    for start_sample in range(0, total_samples, segment_samples):
        end_sample = start_sample + segment_samples
        
        # 获取当前窗口的音频片段
        user_chunk = user_audio_full[start_sample:end_sample]
        system_chunk = system_audio_full[start_sample:end_sample]

        # 如果最后一个片段长度不足，跳过以避免错误（或可以进行填充）
        if user_chunk.shape[0] == 0:
            continue

        # --- 为当前窗口准备模型输入 ---
        user_feats = torch.tensor(feature_extractor(user_chunk, sampling_rate=target_sr)["input_values"][0]).T
        system_feats = torch.tensor(feature_extractor(system_chunk, sampling_rate=target_sr)["input_values"][0]).T
    
        user_input_values = user_feats.unsqueeze(0).transpose(1, 2).to(device)
        system_input_values = system_feats.unsqueeze(0).transpose(1, 2).to(device)
    
        user_padding_mask = torch.ones(1, user_feats.size(0), dtype=torch.bool).to(device)
        system_padding_mask = torch.ones(1, system_feats.size(0), dtype=torch.bool).to(device)

        # --- 对当前窗口进行推理 ---
        with torch.no_grad():
            out = model(
                user_input_values=user_input_values,
                user_padding_mask=user_padding_mask,
                system_input_values=system_input_values,
                system_padding_mask=system_padding_mask,
                look_ahead=args.look_ahead
            )
    
        logits_u, logits_s = out["user_output"], out["system_output"]
        
        # 获取当前窗口的帧级别预测结果
        pred_u_chunk = torch.argmax(logits_u.squeeze(0), dim=-1).cpu().numpy()
        pred_s_chunk = torch.argmax(logits_s.squeeze(0), dim=-1).cpu().numpy()
        
        # 收集结果
        all_preds_u.append(pred_u_chunk)
        all_preds_s.append(pred_s_chunk)
        
        pbar.update(segment_samples)

    pbar.close()

    if not all_preds_u:
        print("[错误] 未生成任何预测结果。请检查音频文件是否过短。")
        return

    # --- 拼接所有窗口的预测结果 ---
    full_pred_u = np.concatenate(all_preds_u)
    full_pred_s = np.concatenate(all_preds_s)

    # --- 生成并打印最终的状态转移矩阵 ---
    frame_duration_s = 1.0 / 12.5 # 每帧80ms

    print("\n## 用户流状态转移矩阵 ##")
    user_transitions = generate_state_transitions(full_pred_u, frame_duration_s)
    for t in user_transitions:
        print(f"  {t['start']:>7s} -> {t['end']:>7s} ({t['duration']:>7s}) | 状态: {t['state_name']}")

    print("\n## 系统流状态转移矩阵 ##")
    system_transitions = generate_state_transitions(full_pred_s, frame_duration_s)
    for t in system_transitions:
        print(f"  {t['start']:>7s} -> {t['end']:>7s} ({t['duration']:>7s}) | 状态: {t['state_name']}")

def parse_args():
    ap = argparse.ArgumentParser(description="使用20秒窗口对DualStreamMimi模型进行单音频推理。")
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
    print("模型加载成功。")

    # --- 加载特征提取器 ---
    print(f"正在加载特征提取器: {args.model_name}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    
    # --- 运行推理 ---
    run_inference(model, args.audio_path, feature_extractor, device, args)

if __name__ == "__main__":
    main()
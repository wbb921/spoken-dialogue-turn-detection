#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for single audio inference using a trained DualStreamMimiModel.
This version processes the audio in 20-second windows, similar to the training setup,
and concatenates the results.
"""

import os
import argparse
import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor
from model.DualStreamMimiModel import DualStreamMimiModel
from tqdm import tqdm

# Mapping from state index to a human-readable name.
# Please adjust these based on your model's classes.
STATE_MAP = {
    0: "Speak",
    1: "Listen",
    2: "Gap",
    3: "Pause",
    4: "Overlap",
}

def generate_state_transitions(prediction_sequence: np.ndarray, frame_duration_s: float) -> list:
    """
    Converts a sequence of frame-level state predictions into a list of state transitions.
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
    Loads a single audio file, performs inference in 20-second windows, 
    prints the final state transition matrix, and saves the result as a NumPy array.
    """
    model.eval()
    print(f"--- Processing audio file: {audio_path} ---")

    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"[ERROR] Could not load audio file: {e}")
        return

    if waveform.shape[0] != 2:
        print(f"[ERROR] Expected a stereo audio file with 2 channels, but got {waveform.shape[0]}.")
        return
        
    user_audio_full, system_audio_full = waveform[0], waveform[1]
    
    target_sr = feature_extractor.sampling_rate
    if sr != target_sr:
        print(f"Resampling audio from {sr} Hz to {target_sr} Hz...")
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        user_audio_full = resampler(user_audio_full)
        system_audio_full = resampler(system_audio_full)

    # --- Windowing settings ---
    window_len_s = 20  # Window length in seconds
    segment_samples = window_len_s * target_sr
    
    all_preds_u = []
    all_preds_s = []

    total_samples = user_audio_full.shape[0]
    
    print(f"Starting inference with {window_len_s}s windows...")
    # Use tqdm for a progress bar
    pbar = tqdm(total=total_samples, unit="samples", desc="Inference Progress")

    # --- Loop through each 20-second window ---
    for start_sample in range(0, total_samples, segment_samples):
        end_sample = start_sample + segment_samples
        
        # Get the current audio chunk
        user_chunk = user_audio_full[start_sample:end_sample]
        system_chunk = system_audio_full[start_sample:end_sample]

        # Skip if the last chunk is empty
        if user_chunk.shape[0] == 0:
            continue

        # --- Prepare model inputs for the current window ---
        user_feats = torch.tensor(feature_extractor(user_chunk, sampling_rate=target_sr)["input_values"][0]).T
        system_feats = torch.tensor(feature_extractor(system_chunk, sampling_rate=target_sr)["input_values"][0]).T
    
        user_input_values = user_feats.unsqueeze(0).transpose(1, 2).to(device)
        system_input_values = system_feats.unsqueeze(0).transpose(1, 2).to(device)
    
        user_padding_mask = torch.ones(1, user_feats.size(0), dtype=torch.bool).to(device)
        system_padding_mask = torch.ones(1, system_feats.size(0), dtype=torch.bool).to(device)

        # --- Run inference on the current window ---
        with torch.no_grad():
            out = model(
                user_input_values=user_input_values,
                user_padding_mask=user_padding_mask,
                system_input_values=system_input_values,
                system_padding_mask=system_padding_mask,
                look_ahead=args.look_ahead
            )
    
        logits_u, logits_s = out["user_output"], out["system_output"]
        
        # Get frame-level predictions for the current window
        pred_u_chunk = torch.argmax(logits_u.squeeze(0), dim=-1).cpu().numpy()
        pred_s_chunk = torch.argmax(logits_s.squeeze(0), dim=-1).cpu().numpy()
        
        # Collect results
        all_preds_u.append(pred_u_chunk)
        all_preds_s.append(pred_s_chunk)
        
        pbar.update(user_chunk.shape[0])

    pbar.close()

    if not all_preds_u:
        print("[ERROR] No predictions were generated. The audio file might be too short.")
        return

    # --- Concatenate results from all windows ---
    full_pred_u = np.concatenate(all_preds_u)
    full_pred_s = np.concatenate(all_preds_s)

    # --- Generate and print the final state transition matrices ---
    frame_duration_s = 1.0 / 12.5 # 80ms per frame

    print("\n## Channel 0 State Transitions ##")
    user_transitions = generate_state_transitions(full_pred_u, frame_duration_s)
    for t in user_transitions:
        print(f"  {t['start']:>7s} -> {t['end']:>7s} ({t['duration']:>7s}) | State: {t['state_name']}")

    print("\n## Channel 1 State Transitions ##")
    system_transitions = generate_state_transitions(full_pred_s, frame_duration_s)
    for t in system_transitions:
        print(f"  {t['start']:>7s} -> {t['end']:>7s} ({t['duration']:>7s}) | State: {t['state_name']}")

    # --- Save the final predictions as a NumPy array ---
    # Stack the predictions into a (2, T) array
    final_states = np.stack([full_pred_u, full_pred_s], axis=0).astype(np.int8)

    # Create the output path
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(args.output_dir, f"{base_name}.npy")
    
    # Save the file
    np.save(output_path, final_states)
    print(f"\n[INFO] Prediction state matrix saved to: {output_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="Run single audio inference with a DualStreamMimi model using 20-second windows.")
    ap.add_argument("--audio_path", required=True, type=str, help="Path to the input stereo .wav file.")
    ap.add_argument("--checkpoint_path", required=True, type=str, help="Path to the trained model checkpoint (.pt) file.")
    ap.add_argument("--output_dir", type=str, default=".", help="Directory to save the output .npy file. Defaults to the current directory.")
    
    ap.add_argument("--model_name", default="kyutai/mimi", help="Base model name from Hugging Face for the feature extractor.")
    ap.add_argument("--num_classes", type=int, default=5, help="Number of output classes for the model.")
    ap.add_argument("--look_ahead", type=int, default=0, help="Look ahead value used during model training.")

    return ap.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Create output directory if it doesn't exist ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Model ---
    print(f"Loading model checkpoint from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint path not found: {args.checkpoint_path}")
        return

    model = DualStreamMimiModel(num_classes=args.num_classes).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    state_dict = checkpoint.get('model_state', checkpoint)
    # Handle models saved with DataParallel
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")

    # --- Load Feature Extractor ---
    print(f"Loading feature extractor: {args.model_name}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    
    # --- Run Inference ---
    run_inference(model, args.audio_path, feature_extractor, device, args)

if __name__ == "__main__":
    main()

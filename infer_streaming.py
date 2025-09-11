#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script for streaming inference with a trained DualStreamMimiModel.
This script simulates a real-time audio stream by processing it in 80ms chunks
while maintaining a sliding 10-second context window.

"""

import os
import argparse
import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor
from models.DualStreamMimiModel import DualStreamMimiModel
from tqdm import tqdm

# Mapping from state index to a human-readable name.
STATE_MAP = {
    0: "Speak",
    1: "Listen",
    2: "Gap",
    3: "Pause",
    4: "Overlap",
}

class StreamProcessor:
    """
    A stateful processor to manage audio buffers and perform streaming inference.
    """
    def __init__(self, model, feature_extractor, device, args):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        self.args = args
        
        # --- Configuration ---
        self.target_sr = feature_extractor.sampling_rate
        self.context_s = 10  # Context window length in seconds
        self.chunk_s = 0.080 # Processing chunk length in seconds (80ms)
        
        # Convert time to number of samples
        self.context_samples = int(self.context_s * self.target_sr)
        self.chunk_samples = int(self.chunk_s * self.target_sr)
        
        # --- State Buffers ---
        self.user_buffer = torch.zeros(self.context_samples, dtype=torch.float32)
        self.system_buffer = torch.zeros(self.context_samples, dtype=torch.float32)
        
        # --- State Tracking ---
        self.user_stream_state = {"current": -1, "start_time": 0.0}
        self.system_stream_state = {"current": -1, "start_time": 0.0}
        self.current_time = 0.0

    def _run_model_on_context(self):
        """Runs the model on the current buffer and returns the last frame's prediction."""
        # Prepare model inputs
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
        
        # We only care about the last frame's prediction, which corresponds to the newest audio chunk
        pred_u = torch.argmax(logits_u.squeeze(0), dim=-1)[-1].item()
        pred_s = torch.argmax(logits_s.squeeze(0), dim=-1)[-1].item()
        
        return pred_u, pred_s

    def _update_and_print_transitions(self, stream_name, stream_state, prediction):
        """Checks for state changes and prints completed segments."""
        if stream_state["current"] == -1: # Initialization
            stream_state["current"] = prediction
        
        if prediction != stream_state["current"]:
            end_time = self.current_time
            start_time = stream_state["start_time"]
            duration = end_time - start_time
            state_id = stream_state["current"]
            state_name = STATE_MAP.get(state_id, f"Unknown State {state_id}")

            # The timestamp corresponds to the *end* of the chunk, so we adjust for printing
            print(f"  [{stream_name}] {(start_time):>7.2f}s -> {(end_time):>7.2f}s ({duration:>6.2f}s) | State: {state_name}")
            
            stream_state["current"] = prediction
            stream_state["start_time"] = end_time

    def process(self, user_chunk, system_chunk):
        """Processes a new 80ms audio chunk and returns the predictions."""
        # Update buffers: roll left and add new data to the end
        self.user_buffer = torch.roll(self.user_buffer, shifts=-self.chunk_samples, dims=0)
        self.user_buffer[-self.chunk_samples:] = user_chunk
        
        self.system_buffer = torch.roll(self.system_buffer, shifts=-self.chunk_samples, dims=0)
        self.system_buffer[-self.chunk_samples:] = system_chunk
        
        # Run inference
        pred_u, pred_s = self._run_model_on_context()
        
        # Update timestamp
        self.current_time += self.chunk_s
        
        # Check and report state transitions for live output
        self._update_and_print_transitions("User Stream  ", self.user_stream_state, pred_u)
        self._update_and_print_transitions("System Stream", self.system_stream_state, pred_s)
        
        return pred_u, pred_s
        
    def flush(self):
        """At the end of processing, prints the last ongoing segment."""
        # Use an invalid state to force-flush the final segment
        self._update_and_print_transitions("User Stream  ", self.user_stream_state, -1)
        self._update_and_print_transitions("System Stream", self.system_stream_state, -1)


def run_streaming_inference(model, audio_path: str, feature_extractor, device, args):
    """
    Loads an audio file and processes it in a streaming fashion.
    """
    print(f"--- Processing audio file: {audio_path} ---")
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"[ERROR] Could not load audio file: {e}")
        return

    if waveform.shape[0] != 2:
        print(f"[ERROR] Expected a stereo audio file, but got {waveform.shape[0]} channels.")
        return
        
    user_audio_full, system_audio_full = waveform[0], waveform[1]
    
    target_sr = feature_extractor.sampling_rate
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        user_audio_full = resampler(user_audio_full)
        system_audio_full = resampler(system_audio_full)

    # Instantiate the stream processor
    stream_processor = StreamProcessor(model, feature_extractor, device, args)
    chunk_samples = stream_processor.chunk_samples
    total_samples = user_audio_full.shape[0]

    # Lists to collect all frame-by-frame predictions for final saving
    all_user_preds = []
    all_system_preds = []

    print("\n## Real-time State Transition Matrix ##")
    pbar = tqdm(total=total_samples, unit="samples", desc="Streaming Inference Progress")

    # Simulate the audio stream by processing 80ms chunks
    for start_sample in range(0, total_samples, chunk_samples):
        end_sample = start_sample + chunk_samples
        if end_sample > total_samples:
            break # Ignore the last partial chunk

        user_chunk = user_audio_full[start_sample:end_sample]
        system_chunk = system_audio_full[start_sample:end_sample]
        
        # Process chunk and collect predictions
        pred_u, pred_s = stream_processor.process(user_chunk, system_chunk)
        all_user_preds.append(pred_u)
        all_system_preds.append(pred_s)

        pbar.update(chunk_samples)
    
    pbar.close()
    stream_processor.flush() # End and print the final segment

    # --- Save the final predictions as a NumPy array ---
    if not all_user_preds:
        print("\n[WARNING] No predictions were generated to save.")
        return

    # Stack the predictions into a (2, T) array
    final_states = np.stack([
        np.array(all_user_preds, dtype=np.int8),
        np.array(all_system_preds, dtype=np.int8)
    ], axis=0)

    # Create the output path
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(args.output_dir, f"{base_name}.npy")
    
    # Save the file
    np.save(output_path, final_states)
    print(f"\n[INFO] Full prediction state matrix saved to: {output_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="Run streaming inference on a DualStreamMimi model using an 80ms chunk and a 10s context window.")
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

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Model ---
    print(f"Loading model checkpoint from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint path not found: {args.checkpoint_path}")
        return
    model = DualStreamMimiModel(num_classes=args.num_classes).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state', checkpoint)
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")

    # --- Load Feature Extractor ---
    print(f"Loading feature extractor: {args.model_name}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    
    # --- Run Streaming Inference ---
    run_streaming_inference(model, args.audio_path, feature_extractor, device, args)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoFeatureExtractor
from model.DualStreamMimiModel import DualStreamMimiModel
from tqdm import tqdm
from torch.utils.data import Subset, random_split
from sklearn.metrics import precision_score, recall_score


def compute_ep_metrics(pred_u, true_u):
    """
    cal EP-Cutoff
    """
    cutoff_events = 0
    total_speech_segments = 0
    endpoint_delays = []
    
    preds = pred_u if isinstance(pred_u, torch.Tensor) else torch.tensor(pred_u)
    trues = true_u if isinstance(true_u, torch.Tensor) else torch.tensor(true_u)
    if preds.dim() == 1: preds = preds.unsqueeze(0)
    if trues.dim() == 1: trues = trues.unsqueeze(0)

    B, T = trues.shape

    USER_SPEECH_LABEL = 0
    USER_VALID_SPEECH_LABEL = [0, 3]
    USER_END_LABEL = 2
    NON_SPEECH_LABELS = {2}
    EDGE_TOLERANCE = 2

    for b in range(B):
        true_seq = trues[b].cpu().numpy()
        pred_seq = preds[b].cpu().numpy()
        
        t = 0
        while t < T - 1:
            if true_seq[t] != USER_SPEECH_LABEL:
                t += 1
                continue

            segment_start_idx = t
            valid_segment_found = False
            
            for j in range(t + 1, T - 1):
                if true_seq[j] == USER_SPEECH_LABEL and true_seq[j+1] == USER_END_LABEL:
                    is_valid_middle = all(label in USER_VALID_SPEECH_LABEL for label in true_seq[t+1:j])
                    if is_valid_middle:
                        segment_end_idx = j
                        total_speech_segments += 1
                        
                        pred_slice = pred_seq[segment_start_idx : segment_end_idx + 1]
                        
                        l_trim, r_trim = 0, 0
                        for i in range(min(EDGE_TOLERANCE, len(pred_slice))):
                            if pred_slice[i] in NON_SPEECH_LABELS: l_trim += 1
                            else: break
                        for i in range(1, min(EDGE_TOLERANCE, len(pred_slice)) + 1):
                            if len(pred_slice) - r_trim - l_trim <= 0: break
                            if pred_slice[-i] in NON_SPEECH_LABELS: r_trim += 1
                            else: break
                        
                        trimmed_pred_slice = pred_slice[l_trim : len(pred_slice) - r_trim]
                        if any(p in NON_SPEECH_LABELS for p in trimmed_pred_slice):
                            cutoff_events += 1
                        
                        future_preds = pred_seq[segment_end_idx:]
                        for offset, p in enumerate(future_preds):
                            if p in NON_SPEECH_LABELS:
                                endpoint_delays.append(offset)
                                break
                        
                        t = segment_end_idx 
                        valid_segment_found = True
                        break
            
            if not valid_segment_found:
                t += 1

    return cutoff_events, total_speech_segments, endpoint_delays


# --------------------------- Helper Function for Labels --------------------------- #
def generate_remaining_time_labels(labels_seq: np.ndarray, frame_duration: float = 0.08, threshold_frame: int = 8) -> np.ndarray:
    """
    Calculates the remaining time for each frame based on the activity label sequence.

    Args:
        labels_seq (np.ndarray): 1D array of activity labels.
        frame_duration (float): Duration of each frame in seconds (e.g., 80ms for MIMI encoder).

    Returns:
        np.ndarray: Remaining time label for each frame (in seconds).
    """
    time_labels = np.zeros_like(labels_seq, dtype=np.int32)
    n = len(labels_seq)
    if n == 0:
        return time_labels

    # Iterate backwards through the sequence
    remaining_frames = 0
    for i in range(n - 1, -1, -1):
        if i < n - 1 and labels_seq[i] == labels_seq[i+1]:
            # If the current frame's activity is the same as the next, increment frame count
            remaining_frames += 1
        else:
            # If the activity changes, reset the counter
            remaining_frames = 0
        
        # The user's original code returns the number of frames, not seconds.
        # To match the original logic, we store the frame count.
        # If you need seconds, uncomment the line below.
        time_labels[i] = min(remaining_frames, threshold_frame)
        # time_labels[i] = remaining_frames * frame_duration # To get time in seconds

    return time_labels

# --------------------------- 1. Dataset Class --------------------------- #
class SpokenWOZDataset(Dataset):
    def __init__(self, audio_dir, label_dir, feature_extractor, target_sampling_rate=24000, window_len=20, overlap_len=10):
        self.audio_paths = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
        assert self.audio_paths, f"No wav files found in {audio_dir}"
        self.label_dir = Path(label_dir)
        self.fe = feature_extractor
        self.tsr = target_sampling_rate

        self.segment_samples = window_len * self.tsr
        self.stride_samples  = overlap_len * self.tsr
        self.segment_labels  = int(window_len * 12.5) # 12.5 Hz label frequency
        self.stride_labels   = int(overlap_len * 12.5)
        self.frame_duration = 1.0 / 12.5 # Corresponds to 80ms per frame

        self.label_paths = {
            Path(p).stem: self.label_dir / (Path(p).stem + ".npy")
            for p in self.audio_paths
        }

        self.index = []
        for wav_path in self.audio_paths:
            key = Path(wav_path).stem
            label_path = self.label_paths[key]
            mat = np.load(label_path)
            num_labels = mat.shape[1]
            num_samples = int(num_labels * self.tsr / 12.5)

            start_sample = 0
            start_label = 0
            while start_sample + self.segment_samples <= num_samples:
                self.index.append((key, start_sample, start_label))
                start_sample += self.stride_samples
                start_label += self.stride_labels

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        key, start_sample, start_label = self.index[idx]
        wav_path = next(p for p in self.audio_paths if Path(p).stem == key)
        waveform, sr = torchaudio.load(wav_path, num_frames=self.segment_samples, frame_offset=start_sample)
        if sr != self.tsr:
            waveform = torchaudio.functional.resample(waveform, sr, self.tsr)

        user_audio   = waveform[0]
        system_audio = waveform[1]

        npy_path = self.label_paths[key]
        mat = np.load(npy_path)
        end_label = start_label + self.segment_labels
        
        # Classification labels
        label_u_np = mat[0, start_label:end_label]
        label_s_np = mat[1, start_label:end_label]
        
        # Generate regression labels for remaining time
        time_label_u_np = generate_remaining_time_labels(label_u_np, self.frame_duration)
        time_label_s_np = generate_remaining_time_labels(label_s_np, self.frame_duration)

        return {
            "key": key,
            "user_audio": user_audio,
            "system_audio": system_audio,
            "label_user": torch.tensor(label_u_np).long(),
            "label_system": torch.tensor(label_s_np).long(),
            "time_label_user": torch.tensor(time_label_u_np).long(),
            "time_label_system": torch.tensor(time_label_s_np).long(),
        }

# --------------------------- 2. Collate Function --------------------------- #
def make_collate_fn(feature_extractor, target_sampling_rate):
    def collate_fn(batches):
        user_audio   = [b["user_audio"] for b in batches]
        system_audio = [b["system_audio"] for b in batches]

        user_feats   = [torch.tensor(feature_extractor(audio, sampling_rate=target_sampling_rate)["input_values"][0]).T
                        for audio in user_audio]
        system_feats = [torch.tensor(feature_extractor(audio, sampling_rate=target_sampling_rate)["input_values"][0]).T
                        for audio in system_audio]

        user_input_values   = pad_sequence(user_feats, batch_first=True, padding_value=0.0).transpose(1, 2)
        system_input_values = pad_sequence(system_feats, batch_first=True, padding_value=0.0).transpose(1, 2)

        user_padding_mask = torch.zeros(user_input_values.shape[0], user_input_values.shape[2], dtype=torch.bool)
        system_padding_mask = torch.zeros(system_input_values.shape[0], system_input_values.shape[2], dtype=torch.bool)

        for i, feat in enumerate(user_feats):
            user_padding_mask[i, :feat.size(0)] = True
        for i, feat in enumerate(system_feats):
            system_padding_mask[i, :feat.size(0)] = True
        
        # Pad classification labels
        label_user = pad_sequence([b["label_user"] for b in batches], batch_first=True, padding_value=-100)
        label_system = pad_sequence([b["label_system"] for b in batches], batch_first=True, padding_value=-100)

        # Pad regression labels
        # We use -100 as a sentinel value to identify padding, which we will ignore during loss calculation.
        time_label_user = pad_sequence([b["time_label_user"] for b in batches], batch_first=True, padding_value=-100)
        time_label_system = pad_sequence([b["time_label_system"] for b in batches], batch_first=True, padding_value=-100)

        return {
            "user_input_values": user_input_values,
            "user_padding_mask": user_padding_mask,
            "system_input_values": system_input_values,
            "system_padding_mask": system_padding_mask,
            "label_user": label_user,
            "label_system": label_system,
            "time_label_user": time_label_user,
            "time_label_system": time_label_system,
        }
    return collate_fn


# --------------------------- VALIDATION FUNCTION [MODIFIED] --------------------------- #
def validate(model, val_loader, cls_criterion, reg_criterion, device, args):
    model.eval()
    total_loss, total_cls_loss, total_reg_loss = 0.0, 0.0, 0.0
    
    # [MODIFIED] Initialize containers for metrics
    all_preds_u, all_trues_u = [], []
    all_preds_s, all_trues_s = [], []
    total_cutoffs, total_segments_for_cutoff, all_endpoint_delays = 0, 0, []

    pbar = tqdm(val_loader, desc="Validating", unit="batch", dynamic_ncols=True, leave=False)
    with torch.no_grad():
        for batch in pbar:
            # --- Move data to device ---
            ui  = batch["user_input_values"].to(device)
            upm = batch["user_padding_mask"].to(device)
            si  = batch["system_input_values"].to(device)
            spm = batch["system_padding_mask"].to(device)
            lu  = batch["label_user"].to(device)
            ls  = batch["label_system"].to(device)
            tlu = batch["time_label_user"].to(device)
            tls = batch["time_label_system"].to(device)

            # --- Forward pass ---
            out = model(
                user_input_values=ui, user_padding_mask=upm,
                system_input_values=si, system_padding_mask=spm,
                look_ahead=args.look_ahead,
            )
            logits_u = out["user_output"]
            logits_s = out["system_output"]
            time_pred_u = out["user_remaining_time"]
            time_pred_s = out["system_remaining_time"]

            # --- Loss Calculation ---
            loss_ce = (cls_criterion(logits_u.view(-1, args.num_classes), lu.view(-1)) + 
                       cls_criterion(logits_s.view(-1, args.num_classes), ls.view(-1))) * 0.5
            
            reg_mask_u = (tlu != -100)
            reg_mask_s = (tls != -100)
            loss_reg_u = cls_criterion(time_pred_u.view(-1, 9), tlu.view(-1))
            loss_reg_s = cls_criterion(time_pred_s.view(-1, 9), tls.view(-1))
            loss_reg = (loss_reg_u + loss_reg_s) * 0.5
            
            loss = loss_ce + args.reg_loss_weight * loss_reg
            total_loss += loss.item()
            total_cls_loss += loss_ce.item()
            total_reg_loss += loss_reg.item()
            
            # --- [MODIFIED] Metric Calculation ---
            pred_u = torch.argmax(logits_u, dim=-1)
            pred_s = torch.argmax(logits_s, dim=-1)

            valid_mask_u = (lu != -100)
            valid_mask_s = (ls != -100)
            
            all_preds_u.extend(pred_u[valid_mask_u].cpu().numpy().tolist())
            all_trues_u.extend(lu[valid_mask_u].cpu().numpy().tolist())
            all_preds_s.extend(pred_s[valid_mask_s].cpu().numpy().tolist())
            all_trues_s.extend(ls[valid_mask_s].cpu().numpy().tolist())
            
            # Calculate endpoint metrics for the user stream
            cutoff_events, total_segments, endpoint_delays = compute_ep_metrics(pred_u, lu)
            total_cutoffs += cutoff_events
            total_segments_for_cutoff += total_segments
            all_endpoint_delays.extend(endpoint_delays)

    # --- Aggregate and Finalize Metrics ---
    avg_loss = total_loss / len(val_loader)
    avg_cls_loss = total_cls_loss / len(val_loader)
    avg_reg_loss = total_reg_loss / len(val_loader)
    
    all_preds = all_preds_u + all_preds_s
    all_trues = all_trues_u + all_trues_s
    
    # Calculate precision and recall
    metrics = {}
    labels_of_interest = [0, 1, 2, 3]
    for label in labels_of_interest:
        metrics[f'p_{label}'] = precision_score(all_trues, all_preds, labels=[label], average="macro", zero_division=0)
        metrics[f'r_{label}'] = recall_score(all_trues, all_preds, labels=[label], average="macro", zero_division=0)
    
    metrics['macro_p'] = precision_score(all_trues, all_preds, labels=labels_of_interest, average="macro", zero_division=0)
    metrics['macro_r'] = recall_score(all_trues, all_preds, labels=labels_of_interest, average="macro", zero_division=0)

    # Calculate endpointing metrics
    metrics['cutoff_rate'] = total_cutoffs / total_segments_for_cutoff if total_segments_for_cutoff > 0 else 0
    metrics['ep_50'] = np.percentile(all_endpoint_delays, 50) * 0.08 * 1000 if all_endpoint_delays else 0 # in ms
    metrics['ep_90'] = np.percentile(all_endpoint_delays, 90) * 0.08 * 1000 if all_endpoint_delays else 0 # in ms

    return avg_loss, avg_cls_loss, avg_reg_loss, metrics


# --------------------------- 3. Train [MODIFIED] --------------------------- #
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    ckpt_dir = Path(f"ckpt/exp_{args.exp_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    TARGET_SR = feature_extractor.sampling_rate

    # ... Dataset and DataLoader setup ...
    full_set = SpokenWOZDataset(args.audio_dir, args.label_dir, feature_extractor, TARGET_SR, args.window_len, args.overlap_len)
    val_ratio = 0.05
    val_len = int(len(full_set) * val_ratio)
    train_len = len(full_set) - val_len
    train_set, val_set = random_split(full_set, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=make_collate_fn(feature_extractor, TARGET_SR), pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=make_collate_fn(feature_extractor, TARGET_SR), pin_memory=(device == "cuda"))
    # ...

    model = DualStreamMimiModel(num_classes=args.num_classes, num_transformer_layers=args.num_transformer_layers).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    cls_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    reg_criterion = nn.MSELoss()

    start_epoch = 1
    best_val_loss = float('inf')
    if args.resume_from and os.path.exists(args.resume_from):
        # ... Resume logic ...
        pass

    # [FIXED] Corrected loop range to respect start_epoch
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss, total_cls_loss, total_reg_loss = 0.0, 0.0, 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}", unit="batch", dynamic_ncols=True)
        for step, batch in enumerate(pbar, 1):
            # --- Move data to device ---
            ui  = batch["user_input_values"].to(device)
            upm = batch["user_padding_mask"].to(device)
            si  = batch["system_input_values"].to(device)
            spm = batch["system_padding_mask"].to(device)
            lu  = batch["label_user"].to(device)
            ls  = batch["label_system"].to(device)
            tlu = batch["time_label_user"].to(device)
            tls = batch["time_label_system"].to(device)

            # --- Forward and Loss Calculation ---
            out = model(
                user_input_values=ui, user_padding_mask=upm,
                system_input_values=si, system_padding_mask=spm,
                look_ahead=args.look_ahead,
            )
            logits_u = out["user_output"]
            logits_s = out["system_output"]
            time_pred_u = out["user_remaining_time"]
            time_pred_s = out["system_remaining_time"]

            loss_ce = (cls_criterion(logits_u.view(-1, args.num_classes), lu.view(-1)) + 
                       cls_criterion(logits_s.view(-1, args.num_classes), ls.view(-1))) * 0.5
            
            reg_mask_u = (tlu != -100)
            reg_mask_s = (tls != -100)
            loss_reg_u = cls_criterion(time_pred_u.view(-1, 9), tlu.view(-1))
            loss_reg_s = cls_criterion(time_pred_s.view(-1, 9), tls.view(-1))
            loss_reg = (loss_reg_u + loss_reg_s) * 0.5

            loss = loss_ce + args.reg_loss_weight * loss_reg

            # --- Backward Pass ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += loss_ce.item()
            total_reg_loss += loss_reg.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "loss_cls": f"{loss_ce.item():.4f}",
                "loss_reg": f"{loss_reg.item():.4f}"
            })

        avg_train_loss = total_loss / len(train_loader)
        avg_train_cls_loss = total_cls_loss / len(train_loader)
        avg_train_reg_loss = total_reg_loss / len(train_loader)
        print(f"\n[Epoch {epoch:02d} Train] ✅ Total Loss: {avg_train_loss:.4f} | CLS Loss: {avg_train_cls_loss:.4f} | REG Loss: {avg_train_reg_loss:.4f}")

        # --- [MODIFIED] Validation ---
        val_loss, val_cls_loss, val_reg_loss, val_metrics = validate(model, val_loader, cls_criterion, reg_criterion, device, args)
        
        # --- [MODIFIED] Logging ---
        print(f"[Epoch {epoch:02d} Valid] 📊 Loss(Total/Cls/Reg): {val_loss:.4f}/{val_cls_loss:.4f}/{val_reg_loss:.4f}")
        print(f"                 🎯 P/R(Macro): {val_metrics['macro_p']:.3f}/{val_metrics['macro_r']:.3f} | "
              f"Cutoff: {val_metrics['cutoff_rate']:.3f} | "
              f"Delay(50/90%): {val_metrics['ep_50']:.0f}ms/{val_metrics['ep_90']:.0f}ms")

        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best_val_loss": best_val_loss}, ckpt_dir / f"epoch_{epoch}.pt")
        # --- Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = ckpt_dir / "best_model.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best_val_loss": best_val_loss}, best_ckpt_path)
            print(f"                 🎉 New best model saved to {best_ckpt_path} with validation loss {val_loss:.4f}")

# ... (parse_args function remains the same) ...

def parse_args():
    # ... (same as previous version) ...
    pass # Placeholder for brevity

if __name__ == "__main__":

    def parse_args():
        ap = argparse.ArgumentParser()
        ap.add_argument("--audio_dir",  default="/path/to/your/audio/directory")
        ap.add_argument("--label_dir",  default="/path/to/your/label/directory")

        ap.add_argument("--model_name", default="kyutai/mimi")
        ap.add_argument("--window_len", type=int, default=20)
        ap.add_argument("--overlap_len", type=int, default=10)
        ap.add_argument("--exp_name",   default="v1")
        ap.add_argument("--num_classes", type=int, default=5)
        ap.add_argument("--batch_size",  type=int, default=32)
        ap.add_argument("--epochs",      type=int, default=50)
        ap.add_argument("--lr",          type=float, default=1e-4)
        ap.add_argument("--look_ahead",  type=int, default=0)
        ap.add_argument("--grad_clip",   type=float, default=1.0)
        ap.add_argument("--num_workers", type=int, default=4)
        ap.add_argument("--resume_from", type=str, default=None)
        ap.add_argument("--num_transformer_layers", type=int, default=1)
        ap.add_argument("--reg_loss_weight", type=float, default=0.5, help="Weight for the remaining time regression loss.")
        return ap.parse_args()

    train(parse_args())
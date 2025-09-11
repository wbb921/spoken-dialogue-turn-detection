import torch
import torch.nn as nn
import torch.nn.functional as F # [NEW] Import functional for ReLU
from transformers import MimiModel
from typing import Optional, Tuple
import math

# CustomMimiModel, create_streaming_memory_mask, PositionalEncoding, 
# and AbsoluteTimePositionalEncoding classes remain the same as you provided.
# ... (insert the other classes here if running as a standalone script) ...

# For completeness, I'll paste the helper classes you provided.

class CustomMimiModel(MimiModel):
    # This class inherits all functionality from transformers.MimiModel
    # We only add one new method below without modifying any existing ones.
    
    def encode_to_hidden_states(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None, # Set as optional
        encoder_past_key_values: Optional[Tuple] = None, # Set as optional and specify type
    ) -> Tuple[torch.Tensor, Optional[Tuple]]: # Modify return type to also return new past_key_values
        """
        Encodes the input audio waveform into pre-quantized continuous hidden states.
        This logic is extracted and modified from the original _encode_frame method.
        
        Returns:
            - hidden_states (torch.Tensor): The encoded hidden states.
            - new_past_key_values (Optional[Tuple]): The updated KV cache for the next iteration.
        """
        # 1. Pass through the initial convolutional encoder
        embeddings = self.encoder(input_values)

        # 2. Pass through the encoder's transformer layers
        encoder_outputs = self.encoder_transformer(
            embeddings.transpose(1, 2), 
            past_key_values=encoder_past_key_values, 
            return_dict=True
        )
        
        # 3. Get continuous hidden states from the transformer output
        hidden_states = encoder_outputs[0].transpose(1, 2)
        
        # Extract the newly generated past_key_values for streaming
        new_past_key_values = encoder_outputs.get("past_key_values")

        # 4. Pass through the downsampling layer if it exists
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        # 5. Return the hidden states and the updated "memory"
        return hidden_states, new_past_key_values

def create_streaming_memory_mask(tgt_len: int, memory_len: int, look_ahead: int, device: str) -> torch.Tensor:
    """
    (New helper function)
    Creates a streaming memory_mask.
    Positions with a value of True are masked (not allowed to attend).
    """
    mask = torch.ones(tgt_len, memory_len, device=device, dtype=torch.bool)
    for t in range(tgt_len):
        # The system at timestep t can see the user's range [0, t + look_ahead]
        visible_end = min(t + look_ahead + 1, memory_len)
        if visible_end > 0:
            mask[t, 0:visible_end] = False
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assumes x shape is (batch, seq_len, d_model)
        # self.pe[:x.size(1)] shape is (seq_len, 1, d_model)
        x = x + self.pe[:x.size(1)].transpose(0,1)
        return self.dropout(x)


class DualStreamMimiModel(nn.Module):
    """
    Dual Stream Model with an added regression head for remaining time prediction.
    """
    def __init__(self,
                 num_classes: int,
                 transformer_dim: int = 512,
                 hidden_dim: int = 512,
                 nhead: int = 4,
                 num_transformer_layers: int = 1,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 mimi_model_name: str = "kyutai/mimi",
                 max_seq_len: int = 5000):

        super().__init__()
        mimi_model = CustomMimiModel.from_pretrained(mimi_model_name)
        self.mimi_encoder = mimi_model

        for param in self.mimi_encoder.parameters():
            param.requires_grad = False

        self.emb_proj = nn.Linear(transformer_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, dropout=dropout, max_len=max_seq_len)

        # User Stream Layers
        user_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.user_self_attention = nn.TransformerEncoder(encoder_layer=user_encoder_layer, num_layers=num_transformer_layers)
        user_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.user_cross_attention = nn.TransformerDecoder(decoder_layer=user_decoder_layer, num_layers=num_transformer_layers)

        # System Stream Layers
        system_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.system_self_attention = nn.TransformerEncoder(encoder_layer=system_encoder_layer, num_layers=num_transformer_layers)
        system_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.system_cross_attention = nn.TransformerDecoder(decoder_layer=system_decoder_layer, num_layers=num_transformer_layers)
        
        # --- Head 1: Classification (Speech Activity) ---
        self.user_classifier = nn.Linear(hidden_dim, num_classes)
        self.system_classifier = nn.Linear(hidden_dim, num_classes)

        # --- [NEW] Head 2: Regression (Remaining Time) ---
        # These heads will map the final hidden state to a single scalar value.
        self.user_classifier_remaining_time = nn.Linear(hidden_dim, 9)
        self.system_classifier_remaining_time = nn.Linear(hidden_dim, 9)


    def forward(self,
                user_input_values: torch.Tensor,
                user_padding_mask: torch.Tensor,
                system_input_values: torch.Tensor,
                system_padding_mask: torch.Tensor,
                look_ahead: int = 5):
        
        with torch.no_grad():
            user_emb, _ = self.mimi_encoder.encode_to_hidden_states(user_input_values, padding_mask=user_padding_mask)
            system_emb, _ = self.mimi_encoder.encode_to_hidden_states(system_input_values, padding_mask=system_padding_mask)

        user_emb = user_emb.transpose(1, 2)
        system_emb = system_emb.transpose(1, 2)

        user_emb = self.emb_proj(user_emb)
        system_emb = self.emb_proj(system_emb)
        
        user_emb = user_emb + self.alpha * self.pos_encoder.pe[:user_emb.size(1)].transpose(0,1)
        system_emb = system_emb + self.alpha * self.pos_encoder.pe[:system_emb.size(1)].transpose(0,1)

        # --- User Path ---
        user_src_mask = nn.Transformer.generate_square_subsequent_mask(user_emb.size(1), device=user_emb.device)
        user_transformer_out = self.user_self_attention(src=user_emb, mask=user_src_mask)

        # --- System Path ---
        system_src_mask = nn.Transformer.generate_square_subsequent_mask(system_emb.size(1), device=system_emb.device)
        system_self_attn_out = self.system_self_attention(src=system_emb, mask=system_src_mask)

        # --- Cross-Attention ---
        device = system_self_attn_out.device

        # (System attends to User)
        tgt_len_sys, mem_len_user = system_self_attn_out.size(1), user_transformer_out.size(1)
        streaming_mask_sys_to_user = create_streaming_memory_mask(tgt_len_sys, mem_len_user, look_ahead, device)
        system_tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len_sys, device=device)
        system_cross_attn_out = self.system_cross_attention(
            tgt=system_self_attn_out, memory=user_transformer_out,
            tgt_mask=system_tgt_mask, memory_mask=streaming_mask_sys_to_user
        )

        # (User attends to System)
        tgt_len_user, mem_len_sys = user_transformer_out.size(1), system_self_attn_out.size(1)
        streaming_mask_user_to_sys = create_streaming_memory_mask(tgt_len_user, mem_len_sys, look_ahead, device)
        user_tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len_user, device=device)
        user_cross_attn_out = self.user_cross_attention(
            tgt=user_transformer_out, memory=system_self_attn_out,
            tgt_mask=user_tgt_mask, memory_mask=streaming_mask_user_to_sys
        )
        
        # --- Head 1: Classification Logits ---
        system_logits = self.system_classifier(system_cross_attn_out)
        user_logits = self.user_classifier(user_cross_attn_out)

        # --- [NEW] Head 2: Remaining Time Prediction ---
        # Pass the final representations through the regression heads.
        # system_time_pred_raw = self.system_regression_head(system_cross_attn_out)
        # user_time_pred_raw = self.user_regression_head(user_cross_attn_out)
        
        # Apply ReLU to ensure the output is non-negative. Squeeze to remove the last dimension of size 1.
        system_remaining_time = self.system_classifier_remaining_time(system_cross_attn_out)
        user_remaining_time = self.user_classifier_remaining_time(user_cross_attn_out)

        return {
            "user_output": user_logits,
            "system_output": system_logits,
            "user_remaining_time": user_remaining_time,    # [NEW]
            "system_remaining_time": system_remaining_time # [NEW]
        }
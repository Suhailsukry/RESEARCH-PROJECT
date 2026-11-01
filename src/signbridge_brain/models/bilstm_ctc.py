"""Stacked BiLSTM + CTC head (primary model)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMCTC(nn.Module):
    """(B, T, D) landmark features -> (B, T, C) CTC log-probabilities. Class 0 = blank."""

    def __init__(self, input_dim: int, num_classes: int, proj_dim: int = 256, hidden_size: int = 192,
                 num_layers: int = 2, dropout: float = 0.3, input_dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(input_dropout),
        )
        self.lstm = nn.LSTM(
            proj_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """x: (B, T, D) padded; lengths: (B,) true lengths. Returns (log_probs (B,T,C), lengths)."""
        total_len = x.size(1)
        h = self.proj(x)
        packed = pack_padded_sequence(h, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=total_len)
        logits = self.classifier(self.out_dropout(out))
        return torch.log_softmax(logits, dim=-1), lengths

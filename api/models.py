"""
models.py — PyTorch model definitions.
Keeping architecture separate from inference logic means this file
can be imported by both the API and training scripts without pulling
in FastAPI or MediaPipe dependencies.
"""

import torch
import torch.nn as nn


class ASLClassifier(nn.Module):
    """
    Bidirectional LSTM with temporal attention pooling for ASL word recognition.

    Architecture (v7):
        Input  : (B, T, NUM_JOINTS * IN_CHANNELS)  — 50 frames × 220 features
        LSTM   : hidden=384, layers=2, bidirectional → (B, T, 768)
        Attn   : learned per-frame weights           → (B, 768)
        Head   : LayerNorm → Linear(768→256) → GELU → Linear(256→num_classes)

    Features per joint per frame: x, y, dx, dy
    Val accuracy on asl100: 69.2%
    """

    def __init__(
        self,
        input_size: int  = 55 * 4,   # NUM_JOINTS * IN_CHANNELS
        hidden_size: int = 384,
        num_layers: int  = 2,
        num_classes: int = 100,
        dropout: float   = 0.4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _  = self.lstm(x)                              # (B, T, H*2)
        attn_w  = torch.softmax(self.attn(out), dim=1)     # (B, T, 1)
        pooled  = (out * attn_w).sum(dim=1)                # (B, H*2)
        return self.classifier(pooled)
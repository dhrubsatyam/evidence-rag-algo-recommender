"""Neural family tagger, built from scratch in PyTorch.

Architecture: token embedding -> masked mean pooling -> MLP head.
Multi-label output trained with BCEWithLogitsLoss. No pretrained
weights are used anywhere in this model.
"""

from __future__ import annotations

import torch
from torch import nn


class FamilyTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len). Mask out padding (id 0) before pooling.
        mask = (input_ids != 0).unsqueeze(-1).float()
        embedded = self.embedding(input_ids) * mask
        pooled = embedded.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.dropout(pooled))

    def predict_proba(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(input_ids))

"""Protein family classification head for a frozen autoregressive model."""

from pathlib import Path

import torch
from torch import nn

from .gru import GRU


class FamilyClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        head_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if head_hidden_dim < 0:
            raise ValueError("head_hidden_dim cannot be negative")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        if head_hidden_dim == 0:
            self.network = nn.Linear(hidden_dim, num_classes)
        else:
            self.network = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, num_classes),
            )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Map hidden states ``(..., hidden_dim)`` to family logits ``(..., C)``."""
        if states.size(-1) != self.hidden_dim:
            raise ValueError(
                f"states must have {self.hidden_dim} features, got {states.size(-1)}"
            )
        return self.network(states)


def load_family_classifier(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[GRU, FamilyClassifier, dict[str, int]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {
        "head_state_dict",
        "head_config",
        "backbone_state_dict",
        "backbone_config",
        "fam_to_idx",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            f"{path} is not a family-classifier checkpoint (missing {sorted(missing)})"
        )

    backbone = GRU(**checkpoint["backbone_config"])
    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    head = FamilyClassifier(**checkpoint["head_config"])
    head.load_state_dict(checkpoint["head_state_dict"])

    backbone = backbone.to(device).eval().requires_grad_(False)
    head = head.to(device).eval().requires_grad_(False)
    return backbone, head, checkpoint["fam_to_idx"]

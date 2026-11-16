"""Protein family classification head for a frozen autoregressive model."""

from pathlib import Path

import torch
from torch import nn

from .gru import GRU


class FamilyClassifier(nn.Module):
    """Predict the protein family from one prefix summary of a frozen backbone.

    The head reads a single hidden state - the backbone's summary of the residues
    seen so far - and returns family logits. Training covers every prefix length
    rather than only finished sequences, so the head stays usable on a sequence
    that is still being written, which is what classifier guidance needs.

    With ``head_hidden_dim=0`` the head is a plain linear probe, which is the
    honest baseline for "how much family information does the frozen model already
    carry?". Anything larger adds one hidden layer.
    """

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
    """Rebuild the frozen backbone and its trained head from one checkpoint.

    The checkpoint written by ``train_classifier.py`` is self-contained, so this
    returns everything guided generation needs: the backbone, the head, and the
    family-name-to-index mapping the head was trained against. Both models come
    back in eval mode with gradients disabled.

    Guidance use::

        backbone, head, fam_to_idx = load_family_classifier(path, device)
        logits, hidden = backbone.step(token, hidden)
        family_logits = head(hidden[-1])          # p(family | prefix so far)
    """
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


def candidate_family_logprobs(
    backbone: GRU,
    head: FamilyClassifier,
    hidden: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """Score the family of every prefix that *could* come next.

    Classifier guidance reweights the next token by how much it moves the sequence
    toward the target family::

        p(x_t = v | x_<t, y) ~ p(x_t = v | x_<t) * p(y | x_<t, v) ** lam

    so the term added to the next-token logits has to be evaluated **per candidate
    token**: it is ``log p(y | x_<t, v)``, not ``log p(y | x_<t)``. Scoring the
    prefix alone gives one number per sequence, which shifts every logit equally
    and leaves the softmax untouched - guidance that silently does nothing.

    This advances the frozen backbone by one step for each candidate at once and
    returns ``(batch, len(candidates), num_classes)`` log-probabilities. The
    backbone's own state is not modified; the caller keeps advancing ``hidden``
    with whichever token it actually samples.

    Usage inside a sampling loop, for target family ``y`` and strength ``lam``::

        logits, hidden = backbone.step(token, hidden)
        scores = candidate_family_logprobs(backbone, head, hidden, candidates)
        guided = logits[:, candidates] + lam * scores[:, :, y]
    """
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape (num_layers, batch, hidden_dim)")
    if candidates.ndim != 1:
        raise ValueError("candidates must have shape (num_candidates,)")
    if candidates.dtype != torch.long:
        raise TypeError("candidates must contain integer token IDs (torch.long)")

    batch = hidden.size(1)
    num_candidates = candidates.numel()
    # Lay the pairs out as [(sequence 0, every candidate), (sequence 1, ...), ...]
    # so that row b * num_candidates + j is sequence b extended by candidate j.
    expanded = hidden.repeat_interleave(num_candidates, dim=1).contiguous()
    tokens = candidates.repeat(batch)

    _, next_hidden = backbone.step(tokens, expanded)
    logits = head(next_hidden[-1])
    return torch.log_softmax(logits, dim=-1).view(batch, num_candidates, -1)

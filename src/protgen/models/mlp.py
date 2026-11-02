import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """Fixed-context autoregressive model using the previous three tokens."""

    context_length = 3

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )
        self.network = nn.Sequential(
            nn.Linear(self.context_length * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return next-token logits aligned with each input position.

        For position ``i``, the logits use only the three tokens immediately before
        target ``i``. The first context is left-padded with ``pad_token_id``.
        """
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence_length)")
        if tokens.dtype != torch.long:
            raise TypeError("tokens must contain integer token IDs (torch.long)")
        if tokens.size(1) == 0:
            raise ValueError("sequence_length must be greater than zero")

        padded = F.pad(tokens, (self.context_length - 1, 0), value=self.pad_token_id)
        contexts = padded.unfold(
            dimension=1, size=self.context_length, step=1
        )
        embedded = self.embedding(contexts).flatten(start_dim=-2)
        return self.network(embedded)
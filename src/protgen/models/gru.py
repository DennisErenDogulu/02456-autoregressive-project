import torch
from torch import nn


class GRU(nn.Module):
    """Autoregressive language model with a unidirectional GRU."""

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return next-token logits aligned with each input position."""
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence_length)")
        if tokens.dtype != torch.long:
            raise TypeError("tokens must contain integer token IDs (torch.long)")
        if tokens.size(1) == 0:
            raise ValueError("sequence_length must be greater than zero")

        outputs, _ = self.gru(self.embedding(tokens))
        return self.output_projection(self.dropout(outputs))

    def step(
        self,
        tokens: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one token per batch item and return logits and updated state."""
        if tokens.ndim != 1:
            raise ValueError("tokens must have shape (batch,)")
        if tokens.dtype != torch.long:
            raise TypeError("tokens must contain integer token IDs (torch.long)")

        outputs, hidden = self.gru(self.embedding(tokens.unsqueeze(1)), hidden)
        logits = self.output_projection(self.dropout(outputs[:, 0, :]))
        return logits, hidden

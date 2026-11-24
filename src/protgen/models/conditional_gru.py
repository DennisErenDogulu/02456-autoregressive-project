import torch
from torch import nn


class ConditionalGRU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        pad_token_id: int = 0,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.num_classes = num_classes
        self.null_label = num_classes
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )
        # One row per family, plus the null row that stands for "no label given".
        self.label_embedding = nn.Embedding(num_classes + 1, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def _check_labels(self, labels: torch.Tensor, batch: int) -> None:
        if labels.ndim != 1:
            raise ValueError("labels must have shape (batch,)")
        if labels.dtype != torch.long:
            raise TypeError("labels must contain integer class IDs (torch.long)")
        if labels.size(0) != batch:
            raise ValueError("labels must have one entry per sequence")
        if int(labels.min()) < 0 or int(labels.max()) > self.null_label:
            raise ValueError(
                f"labels must be in [0, {self.null_label}], where {self.null_label} "
                "is the null label"
            )

    def null_labels(self, batch: int, device: torch.device | str) -> torch.Tensor:
        """The all-null label vector, i.e. what unconditional generation passes."""
        return torch.full((batch,), self.null_label, dtype=torch.long, device=device)

    def drop_labels(
        self,
        labels: torch.Tensor,
        probability: float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if probability == 0:
            return labels
        drop = (
            torch.rand(labels.shape, device=labels.device, generator=generator)
            < probability
        )
        return labels.masked_fill(drop, self.null_label)

    def hidden_states(self, tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence_length)")
        if tokens.dtype != torch.long:
            raise TypeError("tokens must contain integer token IDs (torch.long)")
        if tokens.size(1) == 0:
            raise ValueError("sequence_length must be greater than zero")
        self._check_labels(labels, tokens.size(0))

        embedded = self.embedding(tokens) + self.label_embedding(labels).unsqueeze(1)
        outputs, _ = self.gru(embedded)
        return outputs

    def forward(self, tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return next-token logits aligned with each input position."""
        return self.output_projection(
            self.dropout(self.hidden_states(tokens, labels))
        )

    def step(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 1:
            raise ValueError("tokens must have shape (batch,)")
        if tokens.dtype != torch.long:
            raise TypeError("tokens must contain integer token IDs (torch.long)")
        self._check_labels(labels, tokens.size(0))

        embedded = self.embedding(tokens.unsqueeze(1)) + self.label_embedding(
            labels
        ).unsqueeze(1)
        outputs, hidden = self.gru(embedded, hidden)
        logits = self.output_projection(self.dropout(outputs[:, 0, :]))
        return logits, hidden

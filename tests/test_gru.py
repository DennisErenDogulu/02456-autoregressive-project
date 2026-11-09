import math
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from generate import sample_sequences
from protgen.data import AA
from protgen.models import GRU, MLP
from train import run_epoch


class GRUTests(unittest.TestCase):
    def test_step_matches_full_sequence_forward(self) -> None:
        model = GRU(
            vocab_size=23,
            embedding_dim=8,
            hidden_dim=12,
            num_layers=2,
            dropout=0.2,
        )
        model.eval()
        tokens = torch.randint(0, 23, (2, 6), dtype=torch.long)

        expected = model(tokens)
        hidden = None
        step_logits = []
        for position in range(tokens.size(1)):
            logits, hidden = model.step(tokens[:, position], hidden)
            step_logits.append(logits)

        torch.testing.assert_close(torch.stack(step_logits, dim=1), expected)

    def test_training_epoch_ignores_padding_targets(self) -> None:
        tokens = torch.tensor([[1, 3, 4, 0], [1, 5, 0, 0]], dtype=torch.long)
        targets = torch.tensor([[3, 4, 2, -1], [5, 2, -1, -1]], dtype=torch.long)
        labels = torch.tensor([0, 1], dtype=torch.long)
        lengths = torch.tensor([2, 1], dtype=torch.long)
        batches = DataLoader(
            TensorDataset(tokens, targets, labels, lengths), batch_size=2
        )
        model = GRU(vocab_size=23, embedding_dim=8, hidden_dim=12)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        loss, accuracy = run_epoch(model, batches, torch.device("cpu"), optimizer)

        self.assertTrue(math.isfinite(loss))
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_sampling_supports_gru_and_mlp_with_fixed_lengths(self) -> None:
        models = (
            GRU(vocab_size=23, embedding_dim=8, hidden_dim=12),
            MLP(vocab_size=23, embedding_dim=8, hidden_dim=12, dropout=0.0),
        )
        for model in models:
            with self.subTest(model=type(model).__name__):
                samples = sample_sequences(
                    model,
                    num_samples=2,
                    max_length=5,
                    temperature=1.0,
                    device=torch.device("cpu"),
                    target_lengths=[3, 5],
                )
                self.assertEqual([len(sequence) for sequence in samples], [3, 5])
                self.assertTrue(all(set(sequence) <= set(AA) for sequence in samples))


if __name__ == "__main__":
    unittest.main()

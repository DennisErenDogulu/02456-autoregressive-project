"""Train the fixed-context MLP language model.

Run from the repository root with `uv run python train.py`.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from protgen.data import IGNORE_INDEX, ProteinDataset
from protgen.models import MLP


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, default=ROOT / "data/train.csv")
    parser.add_argument("--val-data", type=Path, default=ROOT / "data/val.csv")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "results/checkpoints/mlp.pt"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.patience < 1:
        parser.error("--patience must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    return args


def run_epoch(
    model: MLP,
    batches: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    with torch.set_grad_enabled(training):
        for tokens, targets, _, lengths in batches:
            max_residue_length = int(lengths.max().item())
            tokens = tokens[:, : max_residue_length + 1].to(
                device, non_blocking=device.type == "cuda"
            )
            targets = targets[:, : max_residue_length + 1].to(
                device, non_blocking=device.type == "cuda"
            )
            valid_targets = targets != IGNORE_INDEX
            token_count = int(valid_targets.sum().item())
            if token_count == 0:
                raise ValueError("A batch contains no next-token targets")

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(tokens)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )

            if training:
                (loss_sum / token_count).backward()
                optimizer.step()

            total_loss += float(loss_sum.detach().item())
            total_tokens += token_count
            correct_tokens += int(
                ((logits.argmax(dim=-1) == targets) & valid_targets).sum().item()
            )

    if total_tokens == 0:
        raise ValueError("The data loader produced no next-token targets")
    return total_loss / total_tokens, correct_tokens / total_tokens


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is unavailable")

    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_data = ProteinDataset(args.train_data)
    val_data = ProteinDataset(args.val_data)
    if train_data.vocab != val_data.vocab:
        raise ValueError("Training and validation data use different vocabularies")

    generator = torch.Generator().manual_seed(args.seed)
    pin_memory = device.type == "cuda"
    train_batches = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=pin_memory,
    )
    val_batches = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model_config = {
        "vocab_size": train_data.vocab_size,
        "pad_token_id": train_data.vocab["<PAD>"],
        "embedding_dim": 32,
        "hidden_dim": 256,
        "dropout": 0.1,
    }
    model = MLP(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"device: {device}")
    print(f"training sequences: {len(train_data)}; validation sequences: {len(val_data)}")
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_batches, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_batches, device)
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.1%} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.1%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "training_config": {
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "seed": args.seed,
                    },
                },
                args.checkpoint,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping after epoch {epoch}")
                break

    print(
        f"best validation loss: {best_val_loss:.4f} at epoch {best_epoch}; "
        f"checkpoint: {args.checkpoint}"
    )


if __name__ == "__main__":
    main()

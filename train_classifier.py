import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from protgen.data import ProteinDataset
from protgen.models import GRU, FamilyClassifier


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone",
        type=Path,
        default=ROOT / "results/checkpoints/gru_h256_l2.pt",
        help="frozen language-model checkpoint written by train.py",
    )
    parser.add_argument("--train-data", type=Path, default=ROOT / "data/train.csv")
    parser.add_argument("--val-data", type=Path, default=ROOT / "data/val.csv")
    parser.add_argument("--test-data", type=Path, default=ROOT / "data/test.csv")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=128,
        help="hidden width of the head; 0 trains a plain linear probe",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--min-prefix",
        type=int,
        default=0,
        help="shortest prefix (in residues) the head is trained and scored on",
    )
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
    if args.head_hidden_dim < 0:
        parser.error("--head-hidden-dim cannot be negative")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if args.min_prefix < 0:
        parser.error("--min-prefix cannot be negative")
    return args


def load_backbone(path: Path, device: torch.device) -> tuple[GRU, dict]:
    """Load a GRU language model from a train.py checkpoint and freeze it."""
    if not path.is_file():
        raise SystemExit(
            f"No checkpoint found at {path}; train the language model first with "
            "`uv run python train.py --model gru`"
        )
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not {
        "model_config",
        "model_state_dict",
    }.issubset(checkpoint):
        raise SystemExit(f"{path} is not a compatible model checkpoint")
    if checkpoint.get("model_type") != "gru":
        raise SystemExit(
            f"{path} holds a {checkpoint.get('model_type')} model. The head needs the "
            "recurrent state of a GRU, which summarises the whole prefix; the MLP only "
            "ever sees three residues. Train one with `python train.py --model gru`."
        )

    config = checkpoint["model_config"]
    backbone = GRU(**config)
    backbone.load_state_dict(checkpoint["model_state_dict"])
    # Frozen in both senses: no gradients, and eval mode so the backbone's own
    # dropout is off and a given prefix always maps to the same state.
    backbone = backbone.to(device).eval().requires_grad_(False)
    return backbone, config


def prefix_mask(
    lengths: torch.Tensor, num_positions: int, min_prefix: int
) -> torch.Tensor:
    positions = torch.arange(num_positions, device=lengths.device)
    return (positions.unsqueeze(0) >= min_prefix) & (
        positions.unsqueeze(0) <= lengths.unsqueeze(1)
    )


def run_epoch(
    backbone: GRU,
    head: FamilyClassifier,
    batches: DataLoader,
    device: torch.device,
    min_prefix: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """One pass over the data, scoring every prefix of every sequence."""
    training = optimizer is not None
    head.train(training)
    total_loss = 0.0
    total_prefixes = 0
    correct = 0

    for tokens, _, labels, lengths in batches:
        non_blocking = device.type == "cuda"
        lengths = lengths.to(device, non_blocking=non_blocking)
        num_positions = int(lengths.max().item()) + 1
        tokens = tokens[:, :num_positions].to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        mask = prefix_mask(lengths, num_positions, min_prefix)
        prefix_count = int(mask.sum().item())
        if prefix_count == 0:
            continue

        with torch.no_grad():
            states = backbone.hidden_states(tokens)

        with torch.set_grad_enabled(training):
            logits = head(states)[mask]
            targets = labels.unsqueeze(1).expand(-1, num_positions)[mask]
            loss_sum = F.cross_entropy(logits, targets, reduction="sum")

            if training:
                optimizer.zero_grad(set_to_none=True)
                (loss_sum / prefix_count).backward()
                optimizer.step()

        total_loss += float(loss_sum.detach().item())
        total_prefixes += prefix_count
        correct += int((logits.detach().argmax(dim=-1) == targets).sum().item())

    if total_prefixes == 0:
        raise ValueError("The data loader produced no prefixes to train on")
    return total_loss / total_prefixes, correct / total_prefixes


@torch.inference_mode()
def collect_predictions(
    backbone: GRU,
    head: FamilyClassifier,
    batches: DataLoader,
    device: torch.device,
    min_prefix: int,
) -> pd.DataFrame:
    """Score every prefix in a split and return one row per prefix."""
    head.eval()
    records = []

    for tokens, _, labels, lengths in batches:
        lengths = lengths.to(device)
        num_positions = int(lengths.max().item()) + 1
        tokens = tokens[:, :num_positions].to(device)
        labels = labels.to(device)

        mask = prefix_mask(lengths, num_positions, min_prefix)
        if not bool(mask.any()):
            continue

        logits = head(backbone.hidden_states(tokens))
        targets = labels.unsqueeze(1).expand(-1, num_positions)
        positions = torch.arange(num_positions, device=device).expand(
            tokens.size(0), -1
        )
        sequence_lengths = lengths.unsqueeze(1).expand(-1, num_positions)

        flat_logits = logits[mask]
        flat_targets = targets[mask]
        nll = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        records.append(
            pd.DataFrame(
                {
                    "prefix_length": positions[mask].cpu().numpy(),
                    "sequence_length": sequence_lengths[mask].cpu().numpy(),
                    "label": flat_targets.cpu().numpy(),
                    "prediction": flat_logits.argmax(dim=-1).cpu().numpy(),
                    "nll": nll.cpu().numpy(),
                }
            )
        )

    return pd.concat(records, ignore_index=True)


def accuracy_by_length(predictions: pd.DataFrame) -> pd.DataFrame:
    correct = predictions["prediction"] == predictions["label"]
    grouped = predictions.assign(correct=correct).groupby("prefix_length")
    return (
        grouped.agg(
            prefixes=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_nll=("nll", "mean"),
        )
        .reset_index()
        .sort_values("prefix_length")
    )


def prefix_report(predictions: pd.DataFrame, families: tuple[str, ...]) -> pd.DataFrame:
    """Summarise accuracy on the finished sequences and on partial ones."""
    predictions = predictions.assign(
        correct=predictions["prediction"] == predictions["label"]
    )
    clean = predictions["prefix_length"] == predictions["sequence_length"]
    fraction = 100 * predictions["prefix_length"] / predictions["sequence_length"].clip(lower=1)
    rows = []

    def summarise(name: str, subset: pd.DataFrame) -> None:
        if subset.empty:
            return
        rows.append(
            {
                "group": name,
                "prefixes": len(subset),
                "accuracy": subset["correct"].mean(),
                "mean_nll": subset["nll"].mean(),
            }
        )

    summarise("clean (full sequence)", predictions[clean])
    summarise("partial (every prefix)", predictions[~clean])
    for low in range(0, 100, 10):
        in_bucket = (fraction >= low) & (fraction < low + 10) & ~clean
        summarise(f"partial {low:>2}-{low + 10:>3}%", predictions[in_bucket])
    for length in (5, 10, 20, 30, 50):
        summarise(
            f"first {length} residues",
            predictions[predictions["prefix_length"] == length],
        )
    for index, family in enumerate(families):
        summarise(
            f"clean, family {family}", predictions[clean & (predictions["label"] == index)]
        )
    summarise("all prefixes", predictions)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is unavailable")

    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_data = ProteinDataset(args.train_data)
    val_data = ProteinDataset(args.val_data)
    if train_data.fam_to_idx != val_data.fam_to_idx:
        raise SystemExit("Training and validation data use different family mappings")

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
        val_data, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory
    )

    backbone, backbone_config = load_backbone(args.backbone, device)
    head_config = {
        "hidden_dim": backbone_config["hidden_dim"],
        "num_classes": len(train_data.fam_to_idx),
        "head_hidden_dim": args.head_hidden_dim,
        "dropout": args.dropout,
    }
    head = FamilyClassifier(**head_config).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    checkpoint_path = args.checkpoint or ROOT / "results/checkpoints/family_classifier.pt"
    frozen = sum(parameter.numel() for parameter in backbone.parameters())
    trainable = sum(parameter.numel() for parameter in head.parameters())
    print(f"device: {device}")
    print(f"backbone: {args.backbone} ({frozen:,} frozen parameters)")
    print(f"head: {trainable:,} trainable parameters")
    print(f"families: {train_data.fam_to_idx}")
    print(
        f"training sequences: {len(train_data)}; validation sequences: {len(val_data)}; "
        f"training prefixes per epoch: {int(sum(train_data.lengths)) + len(train_data):,}"
    )

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, train_accuracy = run_epoch(
            backbone, head, train_batches, device, args.min_prefix, optimizer
        )
        val_loss, val_accuracy = run_epoch(
            backbone, head, val_batches, device, args.min_prefix
        )
        epoch_seconds = time.perf_counter() - epoch_started
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.1%} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.1%} "
            f"epoch_time={epoch_seconds:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "head_state_dict": head.state_dict(),
                    "head_config": head_config,
                    # The frozen backbone travels with the head so guided generation
                    # loads one file and cannot pair the head with a different
                    # language model by accident.
                    "backbone_state_dict": backbone.state_dict(),
                    "backbone_config": backbone_config,
                    "backbone_checkpoint": str(args.backbone),
                    "fam_to_idx": train_data.fam_to_idx,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "training_config": {
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "dropout": args.dropout,
                        "min_prefix": args.min_prefix,
                        "seed": args.seed,
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping after epoch {epoch}")
                break

    print(
        f"best validation loss: {best_val_loss:.4f} (acc {best_val_accuracy:.1%}) "
        f"at epoch {best_epoch}; checkpoint: {checkpoint_path}"
    )

    # Reload the best head, then report how accuracy depends on how much of the
    # sequence the head was allowed to see.
    best = torch.load(checkpoint_path, map_location=device, weights_only=True)
    head.load_state_dict(best["head_state_dict"])

    families = tuple(train_data.fam_to_idx)
    report_path = args.report or ROOT / "results/classifier/prefix_accuracy.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    curves = []
    for split_name, split_path in (("val", args.val_data), ("test", args.test_data)):
        if not split_path.is_file():
            print(f"skipping {split_name}: {split_path} not found")
            continue
        split_data = ProteinDataset(split_path)
        split_batches = DataLoader(split_data, batch_size=args.batch_size, shuffle=False)
        predictions = collect_predictions(
            backbone, head, split_batches, device, args.min_prefix
        )
        report = prefix_report(predictions, families).assign(split=split_name)
        reports.append(report)
        curves.append(accuracy_by_length(predictions).assign(split=split_name))
        print(f"\n{split_name}: {len(split_data)} sequences, {len(predictions):,} prefixes")
        print(
            report[["group", "prefixes", "accuracy", "mean_nll"]].to_string(
                index=False,
                formatters={
                    "accuracy": "{:.1%}".format,
                    "mean_nll": "{:.4f}".format,
                },
            )
        )

    if reports:
        combined = pd.concat(reports, ignore_index=True)
        combined[["split", "group", "prefixes", "accuracy", "mean_nll"]].to_csv(
            report_path, index=False
        )
        print(f"\nWrote prefix-accuracy report to {report_path}")

    if curves:
        curve_path = report_path.with_name("accuracy_by_prefix_length.csv")
        pd.concat(curves, ignore_index=True)[
            ["split", "prefix_length", "prefixes", "accuracy", "mean_nll"]
        ].to_csv(curve_path, index=False)
        print(f"Wrote the accuracy-by-prefix-length curve to {curve_path}")


if __name__ == "__main__":
    main()

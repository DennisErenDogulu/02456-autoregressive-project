"""Sample protein sequences from a trained fixed-context MLP.

Run from the repository root with `uv run python generate.py`.
"""

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from protgen.data import total_tokens
from protgen.models import MLP


ROOT = Path(__file__).resolve().parent
TOKEN_TO_ID = {token: index for index, token in enumerate(total_tokens)}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "results/checkpoints/mlp.pt"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results/sequences/mlp_samples.csv")
    parser.add_argument("--num-samples", type=positive_int, default=256)
    parser.add_argument("--max-length", type=positive_int, default=127)
    parser.add_argument("--temperature", type=positive_float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


@torch.inference_mode()
def sample_sequences(
    model: MLP,
    num_samples: int,
    max_length: int,
    temperature: float,
    device: torch.device,
) -> list[str]:
    pad_id = TOKEN_TO_ID["<PAD>"]
    bos_id = TOKEN_TO_ID["<BOS>"]
    eos_id = TOKEN_TO_ID["<EOS>"]
    context = torch.full(
        (num_samples, model.context_length),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    context[:, -1] = bos_id
    active = [True] * num_samples
    sequences = [[] for _ in range(num_samples)]

    model.eval()
    for _ in range(max_length):
        logits = model(context)[:, -1, :] / temperature
        logits[:, pad_id] = -torch.inf
        logits[:, bos_id] = -torch.inf
        next_tokens = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        next_token_ids = next_tokens.squeeze(1).cpu().tolist()

        for index, token_id in enumerate(next_token_ids):
            if not active[index]:
                continue
            if token_id == eos_id:
                active[index] = False
            else:
                sequences[index].append(total_tokens[token_id])

        context = torch.cat((context[:, 1:], next_tokens), dim=1)
        if not any(active):
            break

    return ["".join(sequence) for sequence in sequences]


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is unavailable")
    if not args.checkpoint.is_file():
        raise SystemExit(
            f"No checkpoint found at {args.checkpoint}; run `uv run python train.py` first"
        )

    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not {
        "model_config",
        "model_state_dict",
    }.issubset(checkpoint):
        raise ValueError(f"{args.checkpoint} is not a compatible MLP checkpoint")

    model_config = checkpoint["model_config"]
    if model_config.get("vocab_size") != len(total_tokens):
        raise ValueError("Checkpoint vocabulary size does not match the project vocabulary")
    if model_config.get("pad_token_id") != TOKEN_TO_ID["<PAD>"]:
        raise ValueError("Checkpoint padding token does not match the project vocabulary")

    model = MLP(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    sequences = sample_sequences(
        model,
        num_samples=args.num_samples,
        max_length=args.max_length,
        temperature=args.temperature,
        device=device,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": [f"mlp_{index:04d}" for index in range(len(sequences))],
            "sequence": sequences,
        }
    ).to_csv(args.out, index=False)
    ended_early = sum(len(sequence) < args.max_length for sequence in sequences)
    print(
        f"Wrote {len(sequences)} sequences to {args.out}; "
        f"{ended_early} ended with EOS before max length, "
        f"mean length {sum(map(len, sequences)) / len(sequences):.1f}"
    )


if __name__ == "__main__":
    main()

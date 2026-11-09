# Autoregressive models

This package contains two next-token language models for protein sequences:

- **MLP**: predicts each next token from a fixed context of the previous three tokens.
- **GRU**: reads tokens left to right and carries a hidden state, so predictions can use the
  preceding sequence.

Both models accept a `torch.long` tensor shaped `(batch, sequence_length)` and return unnormalized
logits shaped `(batch, sequence_length, vocab_size)`. The logits at each position predict the next
token at that position, matching `ProteinDataset.tokens_shifted`. Training ignores targets marked
with `IGNORE_INDEX` so padding does not contribute to the loss.

## Architecture and parameters

| Parameter | MLP default | GRU default | Meaning |
|---|---:|---:|---|
| `vocab_size` | required | required | Number of token IDs; the project dataset has 23, including special tokens. |
| `pad_token_id` | `0` | `0` | Padding ID; its embedding is kept fixed. |
| `embedding_dim` | `32` | `32` | Width of each learned token embedding. |
| `hidden_dim` | `256` | `128` | Width of the MLP hidden layers or GRU recurrent state. |
| `num_layers` | — | `1` | Number of stacked GRU layers; not used by the MLP. |
| `dropout` | `0.1` | `0.1` | Dropout probability. The MLP applies dropout between hidden layers. The GRU applies it to its outputs and between recurrent layers when more than one layer is used. |

The MLP embeds three-token windows, flattens each window, and applies three fully connected
ReLU layers before projecting to vocabulary logits. At the beginning of a sequence, missing
context tokens are left-padded.

The GRU embeds each input token, processes the sequence with a unidirectional `torch.nn.GRU`, and
projects each recurrent output to vocabulary logits. Its `step(tokens, hidden)` method accepts one
token per batch item and returns `(logits, updated_hidden)`, allowing generation to carry recurrent
state forward without reprocessing the whole prefix.

## Training and sampling

Train either model with the shared trainer:

```bash
uv run python train.py --model gru
uv run python train.py --model mlp
```

For example, tune GRU architecture parameters with:

```bash
uv run python train.py --model gru \
  --embedding-dim 32 --hidden-dim 256 --num-layers 2 --dropout 0.1
```

The trainer uses teacher forcing and next-token cross-entropy, selects the checkpoint with the
lowest validation loss, and stops early when validation loss does not improve. See the root
[README training section](../../../README.md#train-and-sample) for all training options and their
defaults. The HPC job uses the two-layer, 256-hidden-unit GRU configuration above; the epoch log
includes elapsed time so its 30-minute walltime can be checked against actual runs.

Generate sequences from a saved GRU checkpoint with:

```bash
uv run python generate.py \
  --checkpoint results/checkpoints/gru.pt \
  --out results/sequences/gru_samples.csv
```

By default, each generated sequence gets a length sampled from the training set. Pass `--free-length`
to let the model stop at `<EOS>` instead, up to `--max-length`. Checkpoints record their model type
and configuration; older MLP checkpoints without a model-type field are loaded as MLPs.

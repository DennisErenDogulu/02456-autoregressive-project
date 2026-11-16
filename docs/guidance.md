# Week 3, Person 2: classifier guidance

The real GRU and Person 1 classifier are available at
`results/checkpoints/gru_h256_l2.pt` and `results/checkpoints/family_classifier.pt`.
Guidance is connected to the existing `generate.py` sampler. No classifier
training or lambda sweep is part of this change.

## Repository trace

- `src/protgen/data/make_data.py`: `ProteinDataset` reads the frozen CSV splits,
  encodes BOS/residues/EOS/PAD, creates shifted next-token targets, and maps
  `AcyP=0`, `HPr=1`, `HU=2`. The 23-token vocabulary is PAD=0, BOS=1, EOS=2,
  then `ACDEFGHIKLMNPQRSTVWY`. Padding targets use `IGNORE_INDEX=-1`.
- `src/protgen/models/mlp.py`: Week 1 `MLP` uses the last three tokens.
  `src/protgen/models/gru.py`: Week 2 `GRU` exposes `step` for recurrent sampling
  and `hidden_states` for classifier inference. There is no Transformer here.
- `train.py`: trains with shifted-target cross-entropy, selects by validation
  loss, and saves `model_type`, `model_config`, and `model_state_dict`.
- `generate.py:main`: reconstructs the model from that checkpoint, optionally
  loads the classifier, chooses training-set lengths unless `--free-length`,
  calls `sample_sequences`, and writes the existing `id,sequence` CSV format.
- `generate.py:sample_sequences`: obtains GRU step logits (or final MLP logits),
  divides by temperature, masks PAD/BOS and optionally EOS, applies guidance,
  then calls `torch.multinomial(torch.softmax(logits, dim=-1), ...)`.
  That softmax is the normalization point.
- `train_classifier.py`: freezes a GRU and fits `FamilyClassifier` on every
  allowed prefix, including BOS by default. `models/classifier.py` provides
  `load_family_classifier`, returning the frozen backbone, head, and label map.
- `config/default.yaml` is empty; generation uses argparse. The local project
  introduction PDF labels classifier guidance as slide 11 and prefix classifier
  training as slide 12. Evaluation scripts and notebooks consume sample CSVs.

## Guidance and classifier interface

For each active sequence and eligible candidate `a`, `protgen.guidance.guide_logits`
constructs `[BOS, ...prefix residues..., a]`. The sampler computes

```text
softmax(generator_logits / temperature + lambda * log p_classifier(y | prefix+a))
```

This equals normalized `p_generator(a | prefix) * p_classifier(y | prefix+a)^lambda`.
Temperature applies to the generator term. Forbidden token logits stay at `-inf`.
Inactive rows are left unchanged and never sent to the classifier. Candidate
inference uses batches of at most 256; each candidate is evaluated on its full
prefix. This trades extra inference work for a simple adapter independent of
the generator's architecture and state.

The explicit adapter `PrefixFamilyClassifier.forward(tokens)` accepts unpadded
`torch.long` tokens shaped `(batch, prefix_length)` on the sampler device, with
BOS and the candidate included. It computes
`head(backbone.hidden_states(tokens)[:, -1, :])`, returning finite raw logits
shaped `(batch, 3)` on that device. `fam_to_idx` comes from the checkpoint;
guidance looks up the target column through that map rather than assuming order.
The real map is `{'AcyP': 0, 'HPr': 1, 'HU': 2}`.

The classifier checkpoint contains `backbone_config`, `backbone_state_dict`,
`head_config`, `head_state_dict`, and `fam_to_idx`, exactly as supplied by Person 1.
Its embedded backbone weights differ from the current generator checkpoint, so
the adapter uses those embedded weights for classification. It never substitutes
the generator's hidden state. No further classifier input is missing.

At `lambda=0`, classifier loading, label lookup, evaluation, and prefix storage
are bypassed. No checkpoint or target family is required. At positive strength,
both are required. Strength must be finite and nonnegative.

PAD/BOS restrictions, training-set length stopping, free-length EOS stopping,
and the maximum length cap are preserved. In free-length mode, EOS is also scored
as an appended candidate. Person 1's current training supervises BOS and residue
prefixes, not EOS states, so classification at an appended EOS extrapolates the
head beyond those supervised positions.

## Commands from the repository root

Unguided generation with the available Week 2 checkpoint:

```bash
uv run python generate.py \
  --checkpoint results/checkpoints/gru_h256_l2.pt \
  --out results/sequences/gru_unguided.csv
```

Guided generation:

```bash
uv run python generate.py \
  --checkpoint results/checkpoints/gru_h256_l2.pt \
  --classifier-checkpoint results/checkpoints/family_classifier.pt \
  --target-family HU --guidance-strength 1 \
  --out results/sequences/gru_HU_lambda1.csv
```

`--lambda 1` is an alias for `--guidance-strength 1`. Append
`--num-samples 2 --free-length --max-length 8 --device cpu` for a short smoke run.
Without `--free-length`, lengths are drawn from `data/train.csv` and EOS is masked.
Existing MLP commands still work with an MLP checkpoint; the unchanged default
`results/checkpoints/mlp.pt` is not present in this checkout.

Tests require only the installed project dependencies and standard-library unittest:

```bash
uv run python -m unittest discover -s tests -p 'test_guidance.py' -v
```

## Verification performed

- `uv run --no-sync python -m unittest discover -s tests -v`:
  11 tests passed. They compare actual next-token probabilities for fixed
  prefixes at zero strength, enforce classifier bypass including checkpoint
  loading, check checkpoint adapter parity, and exercise CLI validation,
  candidate batching, inactive rows, masks, and both stopping modes.
- Deterministic stub: generator probabilities `(0.6, 0.4)` and target-family
  scores `(0.25, 0.75)` give normalized guided probabilities `(1/3, 2/3)` at
  lambda 1 and `(1/7, 6/7)` at lambda 2. Tests verify the actual appended inputs
  and use a permuted label map to detect hard-coded target indices.
- Additional regression against `git show HEAD:generate.py`: all next-token
  probability tensors were bit-for-bit equal across eight steps at lambda 0,
  using the real GRU checkpoint and a small MLP, each in fixed/free-length mode.
- Real checkpoint smoke tests through `uv run --no-sync python generate.py`:
  unguided and HU lambda 1 each wrote two eight-residue free-length samples;
  AcyP lambda 1 with default length sampling wrote lengths 96 and 98. CPU,
  seed 42; files are `/tmp/protgen-unguided-smoke.csv`,
  `/tmp/protgen-guided-smoke.csv`, and `/tmp/protgen-guided-length-smoke.csv`.
  `--no-sync` used the already installed environment. These verify inference and
  CSV output; family enrichment and the full lambda sweep remain evaluation work.

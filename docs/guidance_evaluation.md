# Week 3 evaluation: does classifier guidance raise family agreement?

**Question.** Compared with the same GRU run without guidance, how often does the classifier assign
generated samples to the requested family, and does that come with more duplicates or closer
matches to the training data?

**Scorer caveat for every number below.** The classifier that *guides* the sampler
(`results/checkpoints/family_classifier.pt`) also *scores* the samples. A rise in target fraction is
**agreement with the guiding classifier**. It is not independent evidence of family membership or
function.

## Main result

Guidance strongly increases agreement with the guiding classifier, and the effect saturates by
λ ≈ 2. For all 3 families × 4 strengths × 3 seeds, the target fraction is higher than the matching
unguided baseline. The lowest lower bound of any 95% interval for the change is +22.8 pp.

| Target | Baseline (λ=0) | λ=0.5 | λ=1 | λ=2 | λ=4 |
|---|---|---|---|---|---|
| AcyP | 30–39% | 66–73% | 87–89% | 96–99% | 99–100% |
| HPr | 26–31% | 60–63% | 83–86% | 92–96% | 97–98% |
| HU | 34–41% | 75–79% | 94–96% | 99.6–100% | 100% |

(range over seeds 42, 43, 44; n = 256 per cell, nothing excluded.) The unguided GRU is not an even
split between families, so each family is compared with its own baseline.

**What it costs (descriptive, one generator, 3 seeds):**

- **Validity and uniqueness:** unchanged. All 13 sample sets per seed have `frac_valid` = 1.000 and
  `frac_unique` = 1.000, so there are **no exact duplicate samples** at any strength.
- **Similarity to training data: depends on family.** For HU and HPr, the median nearest-training
  identity rises with λ, from 0.53 to about 0.65 (HU at λ ≥ 2) and 0.64–0.67 (HPr at λ = 4). At HU λ = 2 and 4, every
  per-seed bootstrap interval for the change excludes zero. For AcyP it moves slightly the other way
  (0.53 → about 0.48–0.49). The README cites 0.70 for the real-protein control, which comes from the
  test-derived `control_real`, which I did not use. So strongly guided HU/HPr samples sit
  closer to the training set than unguided ones, though below that reference.
- **Exploratory, not in the CSV (seed 42, first 80 samples per set):** mean pairwise identity
  *within* a set goes from 0.23 (baseline) to 0.33 (HU λ=1), 0.45 (HU λ=4), 0.35 (HPr λ=4) and
  0.35 (AcyP λ=4). Samples are all distinct but more alike each other. Guidance concentrates the
  generator on a narrower, more similar set, even without exact duplicates.

Full per-run table (counts / 256, change vs same-seed baseline in percentage points, existing
evaluator metrics):

| Target | λ | Assigned to target / 256 (seeds 42, 43, 44) | Change vs baseline, pp (seeds 42, 43, 44) | frac_valid | frac_unique | Median nearest-train identity (seeds 42, 43, 44) |
|---|---|---|---|---|---|---|
| AcyP | 0 | 101, 78, 77 | baseline | 1.000 | 1.000 | 0.530, 0.531, 0.529 |
| AcyP | 0.5 | 186, 170, 182 | +33.2, +35.9, +41.0 | 1.000 | 1.000 | 0.492, 0.490, 0.480 |
| AcyP | 1 | 228, 222, 226 | +49.6, +56.2, +58.2 | 1.000 | 1.000 | 0.484, 0.477, 0.480 |
| AcyP | 2 | 250, 245, 254 | +58.2, +65.2, +69.1 | 1.000 | 1.000 | 0.490, 0.485, 0.490 |
| AcyP | 4 | 255, 254, 254 | +60.2, +68.8, +69.1 | 1.000 | 1.000 | 0.489, 0.490, 0.495 |
| HPr | 0 | 67, 74, 79 | baseline | 1.000 | 1.000 | 0.530, 0.531, 0.529 |
| HPr | 0.5 | 161, 155, 159 | +36.7, +31.6, +31.2 | 1.000 | 1.000 | 0.515, 0.520, 0.500 |
| HPr | 1 | 218, 221, 213 | +59.0, +57.4, +52.3 | 1.000 | 1.000 | 0.510, 0.535, 0.524 |
| HPr | 2 | 243, 245, 235 | +68.8, +66.8, +60.9 | 1.000 | 1.000 | 0.610, 0.594, 0.609 |
| HPr | 4 | 249, 251, 249 | +71.1, +69.1, +66.4 | 1.000 | 1.000 | 0.665, 0.667, 0.637 |
| HU | 0 | 88, 104, 100 | baseline | 1.000 | 1.000 | 0.530, 0.531, 0.529 |
| HU | 0.5 | 197, 203, 193 | +42.6, +38.7, +36.3 | 1.000 | 1.000 | 0.574, 0.573, 0.556 |
| HU | 1 | 241, 246, 240 | +59.8, +55.5, +54.7 | 1.000 | 1.000 | 0.614, 0.603, 0.599 |
| HU | 2 | 256, 255, 255 | +65.6, +59.0, +60.5 | 1.000 | 1.000 | 0.652, 0.641, 0.646 |
| HU | 4 | 256, 256, 256 | +65.6, +59.4, +60.9 | 1.000 | 1.000 | 0.654, 0.652, 0.659 |

The baseline row repeats for the three families because the same unguided run is the comparator. The
CSV has Wilson intervals for each fraction, Newcombe intervals for each change, bootstrap intervals for
the identity change, and the mean target probability.

**Invalid or unscored samples.** None occurred (every `n_unscored` is 0). The pipeline counts an
invalid or empty sample in the denominator as *not* assigned to the target, and reports it in
`n_unscored`; `target_frac_of_scored` gives the fraction among valid samples.

## Fairness checks

- Every guided run is `matched` to the same-seed baseline on generator checkpoint
  (`gru_h256_l2.pt`), temperature (1), training-length mode, requested and actual n (256), seed and
  device (cuda). The script flags any difference in `status` and never pools seeds.
- Seeds are not pooled: results are per seed, and a guided run is only compared with the baseline of
  its own seed. All three seeds agree in direction and are close in size.
- Data used: `data/train.csv` as the novelty reference and `data/val.csv` for classifier sanity checks only.
  `data/test.csv` and the test-derived `control_real` were not read.
- Classifier scoring (argmax at the last residue of `[BOS, residues…]`) gives 99.55% accuracy on
  `data/val.csv`, the same as the "clean (full sequence)" row of `results/classifier/prefix_accuracy.csv`.
  It matches the guidance adapter's output.

## Provenance of the runs

- `gru_unguided.csv` (seed 42) was made before this evaluation, and the CSV carries no metadata. Regenerating
  with the defaults on my GPU reproduces 255 of 256 rows (0 on CPU), so its device is inferred as
  cuda. The seed 43/44 baselines were generated with the same defaults.
- The 12 seed-42 guided files and all seed 43/44 files were **generated by me with the existing
  `generate.py`** (no sampler change, no retraining). `gru_HU_lambda1.csv` previously held a
  16-sample smoke run; it is now the full n = 256 run (the smoke version is in git history, commit c3f0b21).
  Settings are in [eval/guidance_runs.csv](../eval/guidance_runs.csv).

## Reproduce

```bash
# generation (seed 42 shown; seeds 43 and 44 add _seed43 / _seed44 to the file names)
uv run python generate.py --checkpoint results/checkpoints/gru_h256_l2.pt \
  --num-samples 256 --temperature 1 --seed 42 --device cuda --out results/sequences/gru_unguided.csv
for FAM in AcyP HPr HU; do for L in 0.5 1 2 4; do
  uv run python generate.py --checkpoint results/checkpoints/gru_h256_l2.pt \
    --classifier-checkpoint results/checkpoints/family_classifier.pt \
    --target-family $FAM --guidance-strength $L --num-samples 256 \
    --temperature 1 --seed 42 --device cuda --out results/sequences/gru_${FAM}_lambda${L}.csv
done; done

# existing evaluator, once per sample CSV (CPU, no --fold)
uv run python eval/eval.py --samples results/sequences/gru_unguided.csv    # and each guided CSV

# analysis: reads eval/guidance_runs.csv, results/eval/<run>/eval.csv, family_classifier.pt
uv run python eval/guidance_agreement.py
```

Outputs, in `results/guidance_eval/` (git-ignored, so commit with `git add -f`):

- `guidance_agreement.csv`: one row per family × λ × seed, 45 rows (36 guided and 9 baseline)
- `guidance_agreement.png`: target-family agreement against λ, one panel per family

If a planned run is missing, the script lists it as `missing` and prints its `generate.py` command.

## Limitations

- **Circularity:** the same classifier guides and scores. Guidance can push samples toward classifier
  quirks (the classifier gives a mean target probability of 0.95–1.00 at λ = 4) without creating real family
  members. Independent checks are needed for any biological claim: ESMFold structure comparison with
  `eval/compare_structures.py` on the guided sets, a separately trained classifier, or HMM profile
  searches.
- **Saturation:** at λ ≥ 2 the HU target fraction is 99.6–100%, so the metric cannot distinguish stronger
  settings. The cost columns (identity, within-set similarity) are the more informative
  ones at high λ.
- One generator checkpoint, one temperature (T = 1) and training-length sampling. Three seeds, with
  256 samples each. The intervals cover sampling noise, are per cell and are not corrected for multiple comparisons.
- `nearest_identity_median` comes from the existing evaluator's top-50 3-mer prefilter. The
  within-set similarity numbers are a small exploratory check, with no CSV or intervals.
- Free-length mode (where the classifier scores EOS states it was never trained on) was not
  evaluated.
- Existing test `test_checkpoint_adapter_matches_person_1_inference` in `tests/test_guidance.py` fails on
  exact tensor equality in this environment (not caused by this work, since `src/` is unchanged). The
  scorer here matched the adapter in my own check, but this should be looked at by Person 2.

## Needs from the team

- **Person 2:** confirm that the settings of `gru_unguided.csv` (device, seed 42) are right; have a look at the failing
  adapter-equality test above.
- **Person 1:** nothing, as long as `family_classifier.pt` is final. Retraining would change both
  guidance and scoring, so every run would need redoing.
- **Team:** decide whether to follow up with an independent check (folding or a second classifier) before
  claiming anything beyond agreement.

# Evaluating generated protein sequences

Tools to check whether the sequences your model generates look like real proteins.
The idea follows the analysis scripts of [DPLM](https://github.com/bytedance/dplm): fold each
sequence with ESMFold and look at the confidence (pLDDT), plus a few cheap sequence statistics.

**You write the model and the sampling code; this folder scores whatever it produces.** The only
thing the two sides agree on is a CSV with a `sequence` column, so nothing here needs to know how
your sequences were made.

## Try it now

`results/sequences/mock_data.csv` is a small sample set that ships with the repo, so the pipeline
can be run the moment `uv sync` has worked - no model needed:

```bash
uv run python eval/eval.py --samples results/sequences/mock_data.csv
```

```
19 sequences, against 16020 in data/train.csv
indexing 16020 reference sequences

setting                  mock_data
n                           19.000
frac_valid                   1.000
frac_unique                  1.000
nearest_identity_median      1.000
```

Those are the right numbers: the mock sequences are copied straight out of `data/train.csv`, so
every one of them is a valid, distinct, **perfect copy of a training protein** -
`nearest_identity_median` is 1. That is the pipeline working, and it is also the failure mode the
metric exists to catch. Your own samples should land well below 1; see the controls below for what
"well below" means.

## The scripts

| Script | Needs GPU | What it does |
|---|---|---|
| `eval.py` | only for `--fold` | **The metrics, one row per sequence:** validity, novelty, uniqueness, and with `--fold` the ESMFold structures and their pLDDT. |
| `make_controls.py` | no | The positive (real test proteins) and negative (the same proteins shuffled) controls, as sample CSVs to score with `eval.py`. |
| `compare_structures.py` | no | Compares the folded structures of several sets: fold (TM-score to real structures), secondary structure, compactness, pLDDT along the chain, structural diversity. |
| `plot_structure.py` | no | PNG + interactive HTML of one predicted structure, coloured by pLDDT. |

All four run anywhere - only `--fold` needs a GPU. Running them on the cluster is `hpc/`, and the
[root README](../README.md#on-the-cluster-dtu-hpc) is the one place that describes it.

The measurements themselves are `protgen.metrics`, so the same number can be had from a notebook
or from your own script without running one of these:

```python
from protgen.metrics import read_reference, read_sequences, sequence_metrics, summarize

samples = read_sequences("results/sequences/mock_data.csv")
reference = read_reference("data/train.csv")
table = sequence_metrics(samples, reference)   # one row per sample
print(summarize(table))                        # one row per set
```

That is all `eval.py` does, plus the command line and writing the two CSVs.

## Input and output

The input is one CSV with a `sequence` column. `eval.py` requires only that column: it derives the
length from the sequence rather than trusting a column, invents ids (`sample_0`, `sample_1`, ...)
when there are none, upper-cases and strips whitespace, and carries any other column through
untouched. Nothing in this folder reads or writes FASTA.

The output directory is named after the input file, so the mock run above writes into
`results/eval/mock_data/`:

| File | What is in it |
|---|---|
| `eval.csv` | One row per sample: its own columns plus one column per metric. This is the result. The sequence is the last column, so the metrics are the part you see first. |
| `summary.csv` | One row per set, every number an aggregate of a column of `eval.csv`, so it is rewritten on every run rather than being a separate measurement. |
| `pdb/` | With `--fold`: one ESMFold structure per valid sample, named after the sample, pLDDT in the B-factor column. |

`eval.csv` still has its `sequence` column, so it can be fed straight back into `eval.py` - after
a fold, for instance, to rescore against a different reference without folding again.

One CSV in, one directory out. A CSV that already carries a `setting` column - several sampling
temperatures concatenated into one file, say - is grouped by it instead, so `is_unique` compares a
sample only with the rest of its own group and `summary.csv` has a row per group. That is the one
case where `setting` appears in `eval.csv`; a single set leaves it out, because the output
directory already says it.

Every sample gets a row, including an empty or invalid one, with `NaN` where a metric does not
apply to it, so `frac_valid` and everything derived from it describe the whole set rather than the
part that happened to be measurable.

## The controls

A single number says very little on its own, and novelty especially: a random sequence is
perfectly novel, and a memorised one scores best on everything else. Two control sets fix the
scale, and they are scored through the same `eval.py` so they are rows in the comparison rather
than special cases in the report.

```bash
uv run python eval/make_controls.py --num 256          # -> results/sequences/control_*.csv
uv run python eval/eval.py --samples results/sequences/control_real.csv
uv run python eval/eval.py --samples results/sequences/control_shuffled.csv
```

| Control | What it is | Why |
|---|---|---|
| `control_real.csv` | real proteins from `data/test.csv`, which no model trained on | the upper end: what the metrics look like for sequences that certainly are proteins |
| `control_shuffled.csv` | the same proteins with their residues permuted | the lower end, with length and composition held fixed, so the gap between the two comes from residue order and nothing else |

Use the same `--num` as the number of generated sequences you are comparing, because some metrics
depend on the size of the set. With the default seed and `--num 256` the two separate clearly:
`nearest_identity_median` is **0.70** for the real control and **0.25** for the shuffled one. A sample set that scores *above* the real control is reproducing training data.

## Reading the results

**Sequence metrics.** Three questions, one column each, plus pLDDT when you fold. The per-sample
column of `eval.csv` is on the left and the `summary.csv` aggregate of it on the right, because
every set-level number here is the mean or median of a column.

| Per sample | Per set | The question it answers |
|---|---|---|
| `valid` | `frac_valid` | **Is it a protein?** Non-empty, and only the 20 standard residues. Every sample gets a row whether or not it is valid, so this is the honest denominator for the rest. |
| `nearest_identity` | `nearest_identity_median` | **Is it new?** Identity to the closest training sequence. Near 1 is copying; low is novel, but a random sequence is novel too, so read it against the controls rather than against 0. `NaN` for an invalid sample. |
| `is_unique` | `frac_unique` | **Is it the same one again?** Whether the sequence appears for the first time in its own set. |
| `mean_plddt` (`--fold`) | `mean_plddt` | **Does it fold?** See the pLDDT paragraph below. |

The reference is `data/train.csv`, which `--reference` changes to any other CSV with a `sequence`
column; measuring novelty against the whole data set would count a generated copy of a held-out
protein as novelty rather than as memorisation.

**pLDDT** (0-100) is ESMFold's own confidence per residue; the table reports the mean over the
protein. Rule of thumb: > 90 very high, 70-90 confident, 50-70 low, < 50 very low (often
disordered or not protein-like). High pLDDT means "folds confidently", not "is functional" and not
"is new": a model that copies the training set also scores well, which is why novelty is reported
beside it.

A good model has high pLDDT, high validity, low exact copying and moderate nearest identity. Plot
the trade-off rather than optimising a single number.

Those three sequence metrics are deliberately few. `protgen.metrics.sequences` has the pieces for a
fourth when a run raises a question they cannot answer - `entropy_bits` and `longest_run` for
repetitive junk such as `AAAAAA...`, `set_metrics` for the k-mer divergence of a whole set from the
reference, `identity` for the mean identity among the samples themselves - and adding a column is
two lines on the table `eval.py` already builds.

## Structures

`--fold` runs ESMFold on every valid sample and needs a GPU and the `eval` dependency group:

```bash
uv run --group eval python eval/eval.py --samples results/sequences/mock_data.csv --fold
```

The 8 GB of ESMFold weights are downloaded once - by `bash hpc/setup_env.sh` on the cluster, or
by the first `--fold` anywhere else - and then read from the cache, without the Hub being
contacted at all. `fold.py` decides where that cache is rather than leaving it to an environment
variable you have to remember: `$BLACKHOLE/huggingface` on the DTU HPC, and `.cache/huggingface`
in this repository anywhere else. To print it:

```bash
uv run python -c "from protgen.metrics.fold import HF_CACHE; print(HF_CACHE)"
```

The [root README](../README.md#where-the-big-files-go) has the rest.

pLDDT says ESMFold is confident, not that it predicts the right fold. Compare the structures of
all sets in one go (CPU only, about half a minute per 1000 structures with 8 workers):

```bash
uv run --group eval python eval/compare_structures.py --out results/structure_comparison \
    --sets real=results/eval/control_real shuffled=results/eval/control_shuffled \
           mine=results/eval/my_samples
```

It writes `structures.csv` (one row per structure), `summary.csv` (one row per set) and
`comparison.png`. The first set (or `--reference`) provides the panel of real structures for the
TM-scores, so list the real control first. Read `frac_same_fold_tm_gt0.5` together with
`mean_pairwise_tm`: every sample can have the right fold while the set as a whole is less diverse
than real proteins (higher mean pairwise TM-score). pLDDT rises with identity to the training data,
so compare sets at similar identity (panel A) before reading a pLDDT difference as a quality
difference.

One structure at a time:

```bash
uv run --group eval python eval/plot_structure.py results/eval/mock_data/pdb/sample_0.pdb
```

The `.png` works anywhere. The `.html` is an interactive 3D view: download it from the cluster and
open it in your browser on your own computer.


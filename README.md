# Autoregressive models of protein sequences

A 02456 Deep Learning project. You build autoregressive sequence models - an MLP with a fixed
context, an RNN / LSTM / GRU, a transformer - on one frozen train/validation/test split of three
protein families, and then measure what they generate.

The repository mainly provides:

- **the data and a dataset class** (`data/`, `src/protgen/data/`), so every model is fitted and
  compared on the same split and the same tokenisation;
- **an evaluation pipeline** (`eval/`, `src/protgen/metrics/`), so "the samples look like
  proteins" becomes a number you can put in a table.

**Remaining**: the models, the training loop and the sampling code. `src/protgen/models/`
and `config/` are empty placeholders to be written.

## Get the code

```bash
git clone https://github.com/Christianhvilshoej/02456-autoregressive-project.git
cd 02456-autoregressive-project
```

Everything below is run from that directory. The data is in the repository, so the clone is all
you need - there is nothing to download separately.

On the cluster, clone it into `$BLACKHOLE` rather than into your home directory, whose quota is
small; see [On the cluster](#on-the-cluster-dtu-hpc).

To work on your own copy, fork it on GitHub first and clone the fork, or point this clone at your
own repository:

```bash
git remote set-url origin https://github.com/<your-user>/<your-repo>.git
```

## Install

The project is managed with [uv](https://docs.astral.sh/uv/). uv installs Python itself as well as
the dependencies, so it is the only thing you need on the machine beforehand:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

`brew install uv`, `pipx install uv` and `winget install --id=astral-sh.uv` work too. Open a new
shell afterwards so that `uv` is on the PATH, then, from the project root:

```bash
uv sync
```

That reads `pyproject.toml` and `uv.lock`, fetches Python 3.13 if the machine has no suitable
interpreter, and creates `.venv/` with the exact locked versions. Prefix a command with `uv run` to
use that environment without activating it:

```bash
uv run python eval/eval.py --samples results/sequences/mock_data.csv
```

If that prints a small table and a results/, the environment works.

**The `eval` group is separate.** To run ESMFold with `eval/eval.py --fold` the eval group is needed:

```bash
uv sync --group eval        # install them
uv run --group eval python eval/eval.py --samples results/sequences/your_data.csv --fold     
```
## The data

Three protein families, balanced, in `data/train.csv`, `data/val.csv` and `data/test.csv` - one row
per protein, columns `sequence,family`. The split is frozen and the files share no sequence, so the
test split is a genuine held-out set; read it once, at the end, and never for model selection.

| Split | Proteins | Per family | Length (min / mean / max) |
|---|---|---|---|
| `train` | 16,020 | 5,340 | 68 / 95.0 / 127 |
| `val` | 2,001 | 667 | 68 / 94.6 / 126 |
| `test` | 2,001 | 667 | 68 / 94.2 / 127 |

### `ProteinDataset`

Use the provided data and data loader by:

```python
from torch.utils.data import DataLoader
from protgen.data import ProteinDataset

train_data = ProteinDataset("data/train.csv")
val_data = ProteinDataset("data/val.csv")
print(train_data)
# ProteinDataset(num_samples=16020, vocab_size=23, num_classes=3, max_length=129)

x, x_shifted, y, length = train_data[0]
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

for x, x_shifted, y, length in train_loader:
    ...
```

| Item | Shape | What it is |
|---|---|---|
| `x` | `(129,)` long | `<BOS>`, the residues, `<EOS>`, then `<PAD>` to `max_length` |
| `x_shifted` | `(129,)` long | `x` shifted left by one: the next-token target, with `IGNORE_INDEX` (`-1`) past the end of the sequence |
| `y` | scalar long | the family, as a class index |
| `length` | int | the number of residues, without `<BOS>`/`<EOS>`/`<PAD>` |


You can also extract whole tensors for the split:

```python
X = train_data.tokens              # (16020, 129) long
Y_gen = train_data.tokens_shifted  # (16020, 129) long, -1 at the padding
Y_fam = train_data.labels          # (16020,) long
```

**The vocabulary** is fixed and can be view at train_data.vocab:
<Insert dictionary printed by data.vocab>

| token | `<PAD>` | `<BOS>` | `<EOS>` | A | C | D | E | F | G | H | I | K | L | M | N | P | Q | R | S | T | V | W | Y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **id** | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 |

**The family ids** are fixed the same way, in `protgen.data.FAMILIES`:

| `y` | 0 | 1 | 2 |
|---|---|---|---|
| **family** | AcyP | HPr | HU |

`train_data.fam_to_idx` is that mapping. `notebooks/introduction.ipynb` runs through all of the
above in a few cells.

## What you build

| Where | What goes there |
|---|---|
| `src/protgen/models/` | the architectures. `mlp.py` is an empty placeholder; add the recurrent and transformer models beside it. Anything you import as `protgen.models...` is picked up without reinstalling, because the package is installed in editable mode. |
| `config/` | run configuration, if you want it. [Hydra](https://hydra.cc) is already a dependency, so `config/default.yaml` can grow into a config group tree; the file is empty, and nothing reads it yet. |
| the project root | your `train.py` / `generate.py`, or notebooks, whichever you prefer. |
| `hpc/train.sh` | the queue, walltime and memory for the training job, and the command that runs it. Fill in the command at the bottom. |

Note that the `eval/` scripts expect your generated sequences to stored as a CSV files at `results/sequences/<name>.csv` like the `mock_data.csv`.

## Evaluating what you generate

See **[`eval/README.md`](eval/README.md)**.
## On the cluster (DTU HPC)
If you are new to the DTU HPC, look at this guide:
https://docs.google.com/document/d/1bRtX87ZD7faG1b5CayN5LKUPcn75xmrWt6jFcuwoGMU/edit?tab=t.0

### Clone under `$BLACKHOLE`

Your home directory holds 30 GB, which this project would fill on its own: the environment is
about 9.5 GB with the CUDA build of torch, and the ESMFold weights another 8 GB. The course has
storage on the **blackhole** scratch filesystem, so you do not have to request a `/work3`
directory yourself. Each of you has your own directory there, and `$BLACKHOLE` points at it:

```bash
echo $BLACKHOLE      # /dtu/blackhole/<your-personal-path>
```

Therefore **DO NOT clone the repo in your home directory**. If you build your code locally with your own git repository and just need the HPC for evaluation push or transfer your samples and clone your version of the github repository:
```bash
# on your laptop
git add results/sequences/my_samples.csv
git commit -m "samples to fold"
git push

# on the login node
cd $BLACKHOLE
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
```
However, if you build everything on the HPC we recommed just cloning this repository once at `$BLACKHOLE`:

```bash
cd $BLACKHOLE
git clone https://github.com/Christianhvilshoej/02456-autoregressive-project.git
cd 02456-autoregressive-project
```

Note that `$BLACKHOLE` is **not backed up**. Push your code to your own GitHub repository
as you go, and copy anything that has to outlive the course - the tables and figures you report
on - somewhere else.

**Everything the cluster needs lives in `hpc/`**, and nothing in `eval/` is specific to it: the
Python entry points run anywhere, the shell scripts in `hpc/` are how you run them here.

| In `hpc/` | How you run it | What it is |
|---|---|---|
| `setup_env.sh` | `bash hpc/setup_env.sh` | one-time setup: installs uv and the environment, downloads the ESMFold weights (~8 GB). Run it from the login node - it only downloads, so it needs the network and no GPU |
| `train.sh` | `bsub < hpc/train.sh` | your training run. Fill in the command at the bottom |
| `eval.sh` | `bsub < hpc/eval.sh` | scores one sample set, structures included - **point `--samples` at your own CSV** |
| `run_eval.sh` | `bash hpc/run_eval.sh <csv>` | one set, in an interactive GPU session (`sxm2sh`, `voltash`, ...) instead of as a batch job |
| `env.sh` | sourced by the others | puts uv on the PATH and uv's package cache under `$BLACKHOLE`; not run directly |

All of it runs from inside that clone. Once, on the login node, and then per run:

```bash
bash hpc/setup_env.sh                                    # once: environment + ESMFold weights

bsub < hpc/train.sh                                      # your training command
bsub < hpc/eval.sh                                       # score one set, ESMFold included
bjobs                                                    # follow them; logs in hpc/logs/

bash hpc/run_eval.sh results/sequences/mock_data.csv      # or one set, interactively
```

Training needs no setup beyond `uv sync`; it is folding that needs `setup_env.sh`. `hpc/eval.sh`
scores the one CSV its `--samples` names, so edit that line and submit it again for each set you
want - your own samples and the two controls from `eval/make_controls.py`. `run_eval.sh` does the
same work interactively, and adds figures of the first five structures; in `eval.sh` that line is
there too, commented out.

Queue, walltime and memory are the `#BSUB` lines at the top of each script rather than
command-line options. Run `bqueues` to see which GPU queues exist; every Nvidia queue works,
because torch on Linux is the CUDA 12.6 build (the PyPI build uses CUDA 13, which has no kernels
for V100s). Each job log starts with the GPU it got.

**The sequence metrics need no GPU**, so only folding has to come here at all:

```bash
uv run python eval/eval.py --samples results/sequences/my_samples.csv   # runs on your laptop
```

## Getting a GPU: interactive or the queue

Two ways to reach a GPU on the cluster, and they suit opposite kinds of work. DTU's own overview
of the GPU nodes, with the current hardware, is worth a read:
<https://www.hpc.dtu.dk/?page_id=2129>.

### Interactive

One command puts you on a GPU node with a shell:

| Command | Node | GPUs |
|---|---|---|
| `voltash` | Tesla V100 | 2, 16 GB each |
| `sxm2sh` | Tesla V100-SXM2 | 4, 32 GB each |
| `a100sh` | Tesla A100 | 2, 40 GB each |

**These nodes are shared.** Several people are logged into one at a time, and every one of them
can see every GPU on it, so the memory you actually get is whatever the others have left. Folding
a large set of long sequences on a busy node is the most reliable way to meet:

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

So look before you start, and pick a device that is idle:

```bash
nvidia-smi                      # memory used / total per device, and whose processes
export CUDA_VISIBLE_DEVICES=1   # the device number from nvidia-smi
```

Set it before the `uv run`, in the same shell.

What interactive nodes are *good* at is the work in the middle of this project: writing code, a
smoke test, and training the models themselves. An MLP, RNN or transformer over 129-token
sequences is small enough to train happily on a shared V100, and you see the result in seconds
instead of waiting for a queue slot.

### The queue

Folding a full sample set is the opposite case - long, memory-hungry and unattended - so it
belongs in a batch job:

```bash
bsub < hpc/eval.sh      # the set that its --samples line names
bjobs                   # follow it; logs in hpc/logs/
```

The `#BSUB -gpu "num=1:mode=exclusive_process"` line in those scripts is what makes this different
from the interactive case: the GPU is yours alone for the job, so no one else's process can appear
on your device halfway through and take the memory you were counting on. Queue, walltime and
memory are the other `#BSUB` lines, as described above.

**Rule of thumb**

| Work | Where |
|---|---|
| writing code, a smoke test on a few samples | interactive |
| training the models in this project | interactive is usually enough |
| `--fold` on a full sample set, or all the controls at once | the queue |

## Layout

```
data/              train.csv, val.csv, test.csv - the frozen split, `sequence,family`
src/protgen/
  data/            ProteinDataset, the vocabulary and the family ids
  models/          empty: your architectures
  metrics/         the measurements eval/ is built from
    sequences.py   reading sample CSVs, validity, identity, k-mer divergence, the summary table
    controls.py    the real and shuffled control sets
    fold.py        ESMFold (needs a GPU and the eval group)
    structures.py  secondary structure, compactness, TM-score
eval/              the entry points, runnable anywhere: eval.py, make_controls.py,
                   compare_structures.py, plot_structure.py. What the numbers mean is in
                   eval/README.md
hpc/               how to run them on the cluster: the #BSUB job scripts, the setup script and the
                   shared settings. logs/ is where a job's .out/.err land
notebooks/         introduction.ipynb - the dataset class in a few cells
config/            empty: your Hydra config, if you use one
results/
  sequences/       sample CSVs (`sequence` column), tracked in git so that they travel to the
                   cluster with the repo. mock_data.csv ships with it
  eval/            gitignored: the metrics, the PDB files from --fold and the figures
```

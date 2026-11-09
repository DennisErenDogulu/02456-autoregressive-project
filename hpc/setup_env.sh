#!/usr/bin/env bash
# One-time setup on the cluster login node:  bash hpc/setup_env.sh
# Installs uv (if missing), the evaluation environment and downloads the ESMFold weights.
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/env.sh

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing the environment (uv sync --group eval) ..."
uv sync --group eval

# LSF opens the job's output files before the script runs, so a missing directory fails the job
# rather than being created by it. The .out/.err files themselves are gitignored.
mkdir -p hpc/logs

# Only download the files: loading the 8 GB model needs more memory than a login node allows.
# The model itself is loaded in the GPU job, which asks for enough memory.
#
# The download goes through the same protgen.metrics.fold.weights_dir() that folding itself
# calls, so the directory filled here is by construction the one every later run reads - rather
# than two pieces of code that agree on a cache only until somebody edits one of them.
echo "Downloading the ESMFold weights (about 8 GB, only the first time) ..."
uv run --group eval python - <<'PY'
from protgen.metrics.fold import weights_dir

print("ESMFold weights are in", weights_dir())
PY

echo
echo "Setup done. Score a set of samples on a GPU with either of:"
echo "    bsub < hpc/eval.sh                                       # one set, as a batch job"
echo "    bash hpc/run_eval.sh results/sequences/mock_data.csv     # one set, in a GPU session"

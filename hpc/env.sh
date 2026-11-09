# Sourced by the other scripts in hpc/. Not run directly. Edit the settings below if your
# cluster differs.

# Optional: modules to load, e.g. EVAL_MODULES="python3/3.12.4". Python is installed by uv,
# and the torch wheels ship their own CUDA libraries, so usually none are needed.
EVAL_MODULES="${EVAL_MODULES:-}"

# The queue, walltime, cores and memory are NOT set here: they are the #BSUB lines at the top of
# hpc/train.sh and hpc/eval.sh. LSF reads those headers as comments before any shell runs, so they
# cannot reference a variable - edit them in the job script itself. Check `bqueues` for the queue
# names in use; torch on Linux is the CUDA 12.6 build (see pyproject.toml), which runs on every
# Nvidia queue: gpuv100, gpua100, gpua10, gpua40, gpul40s.

for module_name in $EVAL_MODULES; do
    module load "$module_name"
done

export PATH="$HOME/.local/bin:$PATH"  # where the uv installer puts uv

# Keep the big caches out of the small home quota: the ESMFold weights (~8 GB) and uv's package
# cache (several GB of torch/CUDA wheels). Both go to the course's blackhole scratch space -
# $BLACKHOLE, which DTU HPC sets per user to /dtu/blackhole/<NN>/<student-number> - and otherwise
# to .cache/ inside this repository. Set HF_HOME / UV_CACHE_DIR to override either one, e.g. to a
# shared read-only copy of the weights (together with HF_HUB_OFFLINE=1, so huggingface_hub does
# not try to write to it).
#
# src/protgen/metrics/fold.py picks the same directory for the weights from inside Python, so
# folding from a notebook or an interactive session does not depend on this file having been
# sourced; what is exported here wins when it has been. UV_CACHE_DIR has no such second home: uv
# runs before any Python of ours does, so running uv outside these scripts does need
# `source hpc/env.sh`.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"  # -P: real path, not via symlinks
# The scripts that source this file run under `set -u`, where a bare $BLACKHOLE would abort them
# in any environment that does not export it - a batch job started from a bare shell, for one.
if [ -d "${BLACKHOLE:-}" ]; then
    CACHE_ROOT="$BLACKHOLE"
else
    CACHE_ROOT="$REPO_DIR/.cache"
    # Downloading 8 GB of ESMFold weights a second time, into a repository that might itself sit in
    # the home quota, is worth a word. A job that did not inherit the login shell's environment is
    # the usual reason $BLACKHOLE is missing here.
    if [ "$(uname -s)" = "Linux" ]; then
        echo "note: \$BLACKHOLE is not set, so the caches go to $CACHE_ROOT instead" >&2
    fi
fi
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

# Loading a .bin checkpoint from a Hub repository with no safetensors copy - facebook/esmfold_v1
# is one - makes transformers download a converted 8 GB in a background thread on every single
# run. `weights_dir` in src/protgen/metrics/fold.py explains why, and avoids it by loading from a
# directory; this covers a from_pretrained written by hand as well.
export DISABLE_SAFETENSORS_CONVERSION="${DISABLE_SAFETENSORS_CONVERSION:-1}"

# uv unpacks every wheel into its cache and then puts the files in .venv. pyproject.toml asks for
# copies, because a checkout in a OneDrive folder cannot take hardlinks; here they can, and the
# CUDA libraries are large enough that the copy doubles the environment - about 9 GB of the same
# bytes twice. Hardlinks need one filesystem for both, so this only applies when the cache and the
# repository are on the same one; otherwise uv falls back to copying without saying so, and the
# warning below is the only hint you would get.
if [ "$(uname -s)" != "Linux" ]; then
    :   # a laptop checkout, where pyproject.toml asks for copies for a reason
elif [ "$(df -P "$UV_CACHE_DIR" | awk 'NR==2 {print $1}')" \
     = "$(df -P "$REPO_DIR"    | awk 'NR==2 {print $1}')" ]; then
    export UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"
else
    echo "note: $UV_CACHE_DIR and $REPO_DIR are on different filesystems, so .venv will be a" >&2
    echo "      full copy of the cache (~9 GB extra). Clone under $CACHE_ROOT to avoid it, or" >&2
    echo "      run 'uv cache clean' once the environment is installed." >&2
fi

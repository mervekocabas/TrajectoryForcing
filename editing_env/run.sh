#!/usr/bin/env bash
# Launch the Trajectory Forcing editing app on a single GPU.
#
# Usage:  ./run.sh [PORT]            (default port 7860, Gradio's standard default)
#   GPU=1 ./run.sh                   (run on GPU 1; defaults to GPU 0 if unset)
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$APP_DIR/.." && pwd)"
cd "$REPO_ROOT"

PORT="${1:-7860}"

# --- pick the GPU to run on ---
# Set GPU=<index> (or export CUDA_VISIBLE_DEVICES yourself) to choose a device.
# On an HTCondor node the assigned GPU is picked up automatically.
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-${_CONDOR_AssignedGPUs:-${GPU_DEVICE_ORDINAL:-}}}}"
if [[ -z "${GPU}" ]]; then
  echo "WARNING: no GPU specified; leaving CUDA_VISIBLE_DEVICES as-is (defaults to GPU 0)." >&2
else
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# --- fail fast if CUDA can't initialize ---
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "ERROR: CUDA is not available in this process — the app would run on CPU (slow)." >&2
  echo "       Check your GPU/driver setup (e.g. 'nvidia-smi') and that CUDA_VISIBLE_DEVICES is valid." >&2
  exit 1
fi
echo "CUDA OK: $(python -c 'import torch;print(torch.cuda.get_device_name(0))')"

# --- proxy + JAX/torch GPU sharing ---
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="$no_proxy"
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # don't let JAX grab all VRAM; leave room for the torch decoder

# Persistent XLA compilation cache so JIT compile is reused across relaunches.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/.jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# --- gradio bind ---
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT="${PORT}"

echo "Warming up model (first boot ~1-2 min); the app URL will be printed once it is"
echo "ready to serve on port ${PORT}. The port is NOT open until then."
exec python -u "$APP_DIR/app.py"

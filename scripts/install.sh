#!/bin/bash
set -euo pipefail

# Create and activate a local virtual environment (override with PYTHON / VENV_DIR).
PYTHON="${PYTHON:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"

# Recreate if missing or incomplete (e.g. an interrupted previous run).
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    rm -rf "${VENV_DIR}"
    "${PYTHON}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
pip install -U pip

# JAX with CUDA 12 GPU support (pulls jaxlib + jax-cuda12 plugin + cuda wheels).
pip install -U "jax[cuda12]==0.4.36"

# JAX training stack.
pip install flax==0.10.4 optax==0.2.5 orbax-checkpoint==0.11.0 chex==0.1.87 \
    ml_dtypes==0.5.1 tensorstore==0.1.76 ml_collections==1.1.0 lpips_j

# Data pipeline + encoder (PyTorch / HuggingFace) and utilities.
pip install torch==2.6.0 torchvision==0.21.0 transformers==5.3.0 fastcluster
pip install "numpy>=2.2" scipy==1.15.3 scikit-learn==1.8.0 pillow==12.1.1 wandb==0.22.0

# Interactive env.
pip install gradio==6.13.0

# FD-loss post-training.
pip install einops diffusers timm muon-optimizer rich tabulate pandas matplotlib opencv-python gdown

#!/bin/bash


set -euo pipefail

BLAS_T="${TF_BLAS_THREADS:-1}"
export OMP_NUM_THREADS="$BLAS_T"
export OPENBLAS_NUM_THREADS="$BLAS_T"
export MKL_NUM_THREADS="$BLAS_T"
export NUMEXPR_NUM_THREADS="$BLAS_T"

CONFIG="${1:-data_prep/configs/choose_encoder.yaml}"
shift || true

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
else
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found and CUDA_VISIBLE_DEVICES is unset; cannot detect GPUs." >&2
        exit 1
    fi
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi

WORLD_SIZE=${#GPUS[@]}
if [[ "${WORLD_SIZE}" -lt 1 ]]; then
    echo "No GPUs detected." >&2
    exit 1
fi

NPROC="$(nproc 2>/dev/null || echo 1)"
# SLURM/Condor sometimes report nproc=1 at script start (delayed cgroup
# setup) even when the job has many CPUs; fall back to the total installed
# CPU count so per-rank cluster_workers doesn't collapse to 1.
[[ "${NPROC}" -le 1 ]] && NPROC="$(nproc --all 2>/dev/null || echo 1)"
DEFAULT_CW=$(( NPROC / WORLD_SIZE ))
[[ "${DEFAULT_CW}" -lt 1 ]] && DEFAULT_CW=1
[[ "${DEFAULT_CW}" -gt 16 ]] && DEFAULT_CW=16
CLUSTER_WORKERS="${CLUSTER_WORKERS:-${DEFAULT_CW}}"

TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="logs/preprocess/${TS}"
mkdir -p "${LOG_DIR}"

echo "Config:           ${CONFIG}"
echo "World size:       ${WORLD_SIZE}  (GPUs: ${GPUS[*]})"
echo "Cluster workers:  ${CLUSTER_WORKERS} per rank  (nproc=${NPROC}; override with CLUSTER_WORKERS=...)"
echo "Log dir:          ${LOG_DIR}"
echo "Extra args:       $*"

PIDS=()
for rank in "${!GPUS[@]}"; do
    gpu_id="${GPUS[$rank]}"
    log_file="${LOG_DIR}/rank_${rank}.log"
    echo "  -> rank ${rank} on cuda:${gpu_id}  (log: ${log_file})"
    (
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        python3 data_prep/imagenet1k_encoder.py \
            --config "${CONFIG}" \
            --device cuda:0 \
            --rank "${rank}" \
            --world_size "${WORLD_SIZE}" \
            --cluster_workers "${CLUSTER_WORKERS}" \
            "$@" \
            >"${log_file}" 2>&1
    ) &
    PIDS+=($!)
done

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAILED=1
    fi
done

if [[ "${FAILED}" -ne 0 ]]; then
    echo "One or more ranks failed; see ${LOG_DIR}/rank_*.log. Skipping finalize." >&2
    exit 1
fi

if [[ "${WORLD_SIZE}" -eq 1 ]]; then
    # The lone rank's encoder already wrote manifest + class_map.json inline
    # (multi-rank suppression only fires when world_size > 1), so the post-hoc
    # --finalize_only would just redo work.
    echo "Single rank — skipping post-hoc --finalize_only (already written inline)."
else
    echo "All ranks finished. Running --finalize_only to write manifest + class_map."
    python3 data_prep/imagenet1k_encoder.py \
        --config "${CONFIG}" \
        --finalize_only \
        2>&1 | tee "${LOG_DIR}/finalize.log"
fi

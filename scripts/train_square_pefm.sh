#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-/home/hikmet/equibot/data/square/pcs}"
PREFIX="${PREFIX:-square_pefm_100d}"
LOG_DIR="${LOG_DIR:-/home/hikmet/equibot/logs/pefm_train}"
NUM_DEMOS="${NUM_DEMOS:-100}"
NUM_EPOCHS="${NUM_EPOCHS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-cuda:0}"
USE_WANDB="${USE_WANDB:-true}"
NUM_WORKERS="${NUM_WORKERS:-8}"

export PYTHONPATH="/home/hikmet/equibot/partial_equivariance:${PYTHONPATH:-}"

env NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}" \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
    HF_HOME="${HF_HOME:-/tmp/huggingface}" \
    conda run --no-capture-output -n lfd python -m pefm.train \
        --config-path /home/hikmet/equibot/partial_equivariance/pefm/configs \
        --config-name square_pefm \
        use_wandb="${USE_WANDB}" \
        prefix="${PREFIX}" \
        log_dir="${LOG_DIR}" \
        data.dataset.path="${DATA_PATH}" \
        data.dataset.num_demos="${NUM_DEMOS}" \
        data.dataset.num_workers="${NUM_WORKERS}" \
        training.batch_size="${BATCH_SIZE}" \
        training.num_epochs="${NUM_EPOCHS}" \
        device="${DEVICE}" \
        "$@"

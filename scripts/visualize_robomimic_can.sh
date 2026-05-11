#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-/home/hikmet/equibot/data/robomimic/can/ph/demo_v15.hdf5}"
OUT_DIR="${OUT_DIR:-/home/hikmet/equibot/data/robomimic/can/ph/visual_compare}"
STEPS="${STEPS:-0,40,80}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
NUM_DEMOS="${NUM_DEMOS:-1}"
NUM_POINTS="${NUM_POINTS:-256}"
CAM_RESOLUTION="${CAM_RESOLUTION:-128}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"

env NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}" \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
    conda run -n lfd python -m equibot.envs.sim_robosuite.visualize_robomimic_pc \
        --dataset "${DATASET}" \
        --task can \
        --out_dir "${OUT_DIR}" \
        --num_demos "${NUM_DEMOS}" \
        --steps "${STEPS}" \
        --num_samples "${NUM_SAMPLES}" \
        --num_points "${NUM_POINTS}" \
        --cam_resolution "${CAM_RESOLUTION}" \
        --camera_name "${CAMERA_NAME}"

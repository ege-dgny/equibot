#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-/home/hikmet/equibot/data/robomimic/can/ph/demo_v15.hdf5}"
OUT_DIR="${OUT_DIR:-/home/hikmet/equibot/data/can}"
NUM_DEMOS="${NUM_DEMOS:-100}"
NUM_POINTS="${NUM_POINTS:-256}"
CAM_RESOLUTION="${CAM_RESOLUTION:-256}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"

env NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}" \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
    conda run -n lfd python -m equibot.envs.sim_robosuite.convert_robomimic \
        --dataset "${DATASET}" \
        --task can \
        --data_out_dir "${OUT_DIR}" \
        --num_demos "${NUM_DEMOS}" \
        --num_points "${NUM_POINTS}" \
        --cam_resolution "${CAM_RESOLUTION}" \
        --camera_name "${CAMERA_NAME}"

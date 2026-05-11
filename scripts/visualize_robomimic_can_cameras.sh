#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-/home/hikmet/equibot/data/robomimic/can/ph/demo_v15.hdf5}"
OUT_DIR="${OUT_DIR:-/home/hikmet/equibot/data/robomimic/can/ph/camera_views}"
DEMO="${DEMO:-demo_0}"
TIMESTEP="${TIMESTEP:-40}"
CAM_RESOLUTION="${CAM_RESOLUTION:-256}"

env NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}" \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
    conda run -n lfd python -m equibot.envs.sim_robosuite.visualize_robomimic_cameras \
        --dataset "${DATASET}" \
        --task can \
        --out_dir "${OUT_DIR}" \
        --demo "${DEMO}" \
        --timestep "${TIMESTEP}" \
        --cam_resolution "${CAM_RESOLUTION}"

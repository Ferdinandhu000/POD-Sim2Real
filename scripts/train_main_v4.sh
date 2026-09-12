#!/usr/bin/env bash
set -e
# Run all experiments in yaml_main_v4 or pass custom arguments
# Usage:
#   # Run all v4 configs sequentially on GPU 0:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_main_v4.sh
#   # Or multi-GPU parallel execution across GPUs 0,1,2,3:
#   bash scripts/train_main_v4.sh --gpus 0,1,2,3
#   # Or run a single config:
#   bash scripts/train_main_v4.sh --config yaml_main_v4/4_10_triad-afno_depth4_grid2d.yaml --gpus 0

if [ "$#" -eq 0 ]; then
    python -m pod_sim2real.training.train --config-dir yaml_main_v4/
else
    # If user provided --config, do not append --config-dir
    has_config=false
    has_config_dir=false
    for arg in "$@"; do
        if [ "$arg" = "--config" ]; then has_config=true; fi
        if [ "$arg" = "--config-dir" ]; then has_config_dir=true; fi
    done
    if [ "$has_config" = true ] || [ "$has_config_dir" = true ]; then
        python -m pod_sim2real.training.train "$@"
    else
        python -m pod_sim2real.training.train --config-dir yaml_main_v4/ "$@"
    fi
fi


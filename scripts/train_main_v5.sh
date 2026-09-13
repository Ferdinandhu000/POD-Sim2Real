#!/usr/bin/env bash
set -e
# Run all experiments in yaml_main_v5 or pass custom arguments
# Usage:
#   # Run all v5 configs sequentially on GPU 0:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_main_v5.sh
#   # Or multi-GPU parallel execution across GPUs 0,1,2,3:
#   bash scripts/train_main_v5.sh --gpus 0,1,2,3
#   # Or run a single config:
#   bash scripts/train_main_v5.sh --config yaml_main_v5/5_00_pod_res_transformer3d_rank96.yaml --gpus 0

if [ "$#" -eq 0 ]; then
    python -m pod_sim2real.training.train --config-dir yaml_main_v5/
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
        python -m pod_sim2real.training.train --config-dir yaml_main_v5/ "$@"
    fi
fi

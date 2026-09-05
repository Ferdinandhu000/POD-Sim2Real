#!/usr/bin/env bash
set -e
python -m pod_sim2real.training.evaluate --checkpoints-dir best_checkpoints/ "$@"

#!/usr/bin/env bash
set -e
python -m pod_sim2real.training.train --config-dir yaml_operators/ "$@"

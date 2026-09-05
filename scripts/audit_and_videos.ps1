$ErrorActionPreference = "Stop"
python -m pod_sim2real.visualization.audit_dataset --data-root .
python -m pod_sim2real.visualization.render_pairs --data-root . --max-frames 0

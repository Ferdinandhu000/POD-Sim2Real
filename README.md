# POD-Sim2Real

[English](README.md) | [中文说明](README_zh.md)

Sim-to-Real flow forecasting on RealPDEBench NACA0025 hydrofoil data. Contains the Triad-MNO model and standard neural operator baselines.

---

## 1. Environment Setup

```bash
# 1. Create and activate conda environment
conda create -n realpde python=3.10 -y
conda activate realpde

# 2. Install PyTorch matching your server CUDA version (e.g. CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install project dependencies and register package
pip install -r requirements.txt
pip install -e .
```

---

## 2. Data Preparation

Place the official RealPDEBench Foil dataset under `data/foil/`:

```text
data/
└── foil/
    ├── hf_dataset/
    │   ├── real/                       # Contains dataset_info.json, state.json, *.arrow
    │   ├── numerical/                  # Contains dataset_info.json, state.json, *.arrow
    │   ├── train_index_real.json
    │   ├── val_index_real.json
    │   ├── test_index_real.json
    │   ├── train_index_numerical.json
    │   ├── val_index_numerical.json
    │   └── test_index_numerical.json
    ├── remain_params_real.json
    ├── in_dist_test_params_real.json
    └── out_dist_test_params_real.json
```

> **Note**: The pipeline rigidly cuts all trajectories to the first 2000 frames (`prefix_frames: 2000`) across train/val/test.

---

## 3. Training

### Option A: Batch Training (Recommended)
```bash
# 1. Train all 10 main comparison models (sequentially):
python -m pod_sim2real.training.train --config-dir yaml_main/
# or: bash scripts/train_main.sh

# 2. Train all 6 ablation models (sequentially):
python -m pod_sim2real.training.train --config-dir yaml_ablation/
# or: bash scripts/train_ablation.sh
```

### Option B: Train a Single Model
```bash
python -m pod_sim2real.training.train --config yaml_main/09_triad-mno_config.yaml
```

---

## 4. Checkpoints & Outputs

- **Runtime Outputs**: Saved in isolated folders under `artifacts/runs/<yaml_name>/` (e.g. `artifacts/runs/09_triad-mno_config/`), including intermediate checkpoints and logs.
- **Best Checkpoint Archiving**: When early stopping occurs (patience = 10), the best validation epoch (e.g. stopped at 53 -> best epoch 43) is automatically copied to `best_checkpoints/<yaml_name>/`:
  - `best.pt`: Best fine-tuned real weights
  - `best_sim.pt`: Best pre-trained sim weights
  - `train.log`: Training log
  - `config.yaml`: Experiment configuration
  - `pod_u.npz`, `pod_v.npz`: POD orthogonal basis matrices
  - `info.txt`: Documents `best_epoch`, early stopping details, and validation metrics

---

## 5. One-Click Evaluation & Excel Report

After training, evaluate all models archived in `best_checkpoints/`:

```bash
bash scripts/evaluate.sh
# or: python -m pod_sim2real.training.evaluate --checkpoints-dir best_checkpoints/
```

This evaluates `seen`, `in_dist`, `out_dist`, and `all` test splits, and generates:
- `evaluation_results_YYYYMMDD_HHMMSS.xlsx`:
  - `Summary`: Formatted table matching paper Table 1 (MSE, Rel-L2, Vorticity MSE, Vorticity Rel-L2).
  - `Detailed_Subsets`: Detailed breakdown across all subsets.
  - `Per_Channel`: Metrics separated by velocity channels ($u$, $v$).
- `evaluation_results_YYYYMMDD_HHMMSS.json`: Companion raw JSON.
- A summary table printed to the terminal.

---

## 6. Testing

Run unit tests to verify environment and pipeline integrity:
```bash
pytest tests
```

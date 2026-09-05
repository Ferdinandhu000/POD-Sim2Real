# POD-Sim2Real: Sim-to-Real Flow Forecasting with Triad-MNO

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**English**](README.md) | [**简体中文**](README_zh.md)

This repository contains the official implementation of **Triad-MNO** (Triad-Coupled Modal Neural Operator with Continuous Spatial Residual Fields) and a comprehensive Sim-to-Real benchmarking suite on the **RealPDEBench NACA0025 Hydrofoil** dataset.

The framework supports numerical simulation pre-training, experimental PIV fine-tuning, automated early stopping with best-checkpoint archiving, and one-click multi-subset evaluation exporting publication-ready Excel reports.

---

## Key Highlights

1. **Triad-MNO Architecture**:
   - **Hussain-Reynolds Triple Decomposition**: $\mathbf{u}(\mathbf{x}, t) = \overline{\mathbf{U}}(\mathbf{x}) + \widetilde{\mathbf{u}}(\mathbf{x}, t) + \mathbf{u}''(\mathbf{x}, t)$, separating base flow, macro coherent structures (Modes 1–32), and fine-scale turbulent residuals (Modes 33–64).
   - **Continuous Neural Field Decoder**: An implicit coordinate neural field $(x, y) \in [-1, 1]^2 \to \Delta \mathbf{u}(\mathbf{x}, t)$ that decodes micro-modes, avoiding modal submergence and enabling zero-shot super-resolution.
   - **Cross-Scale Triad Attention**: Parameterizes Navier-Stokes quadratic advective triad interactions ($Q_{ijk} a_j a_k$) between macro and micro latent tokens.
   - **Vorticity / Enstrophy Regularizer**: Central finite-difference vorticity loss preserving high-frequency rotational dynamics.
   - **Macro-Backbone Freezing (`freeze_macro: true`)**: Retains universal vortex-shedding frequencies learned from simulation while adapting subgrid boundary layers on experimental data.

2. **Rigid 2000-Frame Cutoff (`prefix_frames: 2000`)**:
   - Truncates all trajectories to the first 2000 frames across training, validation, and testing, strictly preventing temporal data leakage.

3. **Batch Experiment Execution & Model Isolation**:
   - Run an entire directory of YAML configurations with `--config-dir`.
   - Each model runs in an isolated directory (`artifacts/runs/[yaml_name]/`), preventing artifact overwriting.

4. **Automated Best-Checkpoint Archiving**:
   - Early stopping (patience = 10) automatically identifies the true optimal validation epoch (e.g., stop at epoch 53 $\to$ best epoch 43).
   - Packages `best.pt`, `best_sim.pt`, `train.log`, `config.yaml`, `results.json`, `split_manifest.json`, `pod_u.npz`, `pod_v.npz`, and `info.txt` into `best_checkpoints/[yaml_name]/`.

5. **One-Click Evaluation & Excel Export**:
   - Evaluates all models in `best_checkpoints/` across official subsets: `seen`, `in_dist`, `out_dist`, and `all`.
   - Generates timestamped reports `evaluation_results_YYYYMMDD_HHMMSS.xlsx` with `Summary` (matching paper Table 1), `Detailed_Subsets`, and `Per_Channel` sheets.

---

## Model Zoo

### Main Comparison Suite (`yaml_main/`)
| Config | Model | Architecture Type | Description |
|:---|:---|:---|:---|
| `00_unet_config.yaml` | `unet` | Direct Grid | 2D U-Net operating directly on spatial grids |
| `01_fno_config.yaml` | `fno` | Direct Grid | 2D Fourier Neural Operator |
| `02_afno_config.yaml` | `afno` | Direct Grid | 2D Adaptive Fourier Neural Operator |
| `03_itransolver_config.yaml` | `itransolver` | Direct Grid | Spatial Transolver with physics-informed attention |
| `04_pod-unet_config.yaml` | `pod-unet` | Modal ROM | Rank-32 POD basis + 1D U-Net coefficient model |
| `05_pod-fno_config.yaml` | `pod-fno` | Modal ROM | Rank-32 POD basis + 1D FNO coefficient model |
| `06_pod-afno_config.yaml` | `pod-afno` | Modal ROM | Rank-32 POD basis + 1D AFNO coefficient model |
| `07_pod-itransformer_config.yaml` | `pod-itransformer` | Modal ROM | Rank-32 POD basis + Inverted Transformer |
| `08_pod-transolver_config.yaml` | `pod-transolver` | Modal ROM | Rank-32 POD basis + Transolver coefficient model |
| `09_triad-mno_config.yaml` | `triad-mno` | Hybrid Multi-Scale | **Triad-MNO** (Full proposed architecture) |

### Systematic Ablation Suite (`yaml_ablation/`)
| Config | Tested Variation | Description |
|:---|:---|:---|
| `00_triad-mno_full_config.yaml` | Full Triad-MNO | Full proposed model reference |
| `01_triad-mno_no_attn_config.yaml` | w/o Triad Attention | Decoupled macro/micro evolution without cross-attention |
| `02_triad-mno_linear64_config.yaml` | w/o Continuous Field | Replaces neural field decoder with standard linear 64-mode SVD |
| `03_triad-mno_no_vort_config.yaml` | w/o Vorticity Loss | Trains with pure velocity MSE loss (no enstrophy constraint) |
| `04_triad-mno_no_freeze_config.yaml` | w/o Macro Freezing | End-to-end real fine-tuning without freezing macro parameters |
| `05_triad-mno_macro_only_config.yaml` | Macro-Only Baseline | Pure rank-32 macro model ($r_{\mathrm{macro}}=32, r_{\mathrm{micro}}=0$) |

---

## Installation & Server Setup

### 1. Create Conda Environment
```bash
conda create -n realpde python=3.10 -y
conda activate realpde
```

### 2. Install PyTorch with CUDA
Install the matching PyTorch wheel for your server's CUDA driver:

```bash
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1 / 12.4:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Dependencies & Project
```bash
# Install core dependencies (PyArrow, Datasets, OpenPyXL, Pandas, SciPy, etc.)
pip install -r requirements.txt

# Install this package in editable mode
pip install -e .
```

---

## Dataset Layout

Download the official **RealPDEBench Foil** dataset from Hugging Face and place it in the `data/` directory:

```text
data/
└── foil/
    ├── hf_dataset/
    │   ├── real/                       # dataset_info.json, state.json, *.arrow
    │   ├── numerical/                  # dataset_info.json, state.json, *.arrow
    │   ├── train_index_real.json
    │   ├── val_index_real.json
    │   ├── test_index_real.json
    │   ├── train_index_numerical.json
    │   ├── val_index_numerical.json
    │   └── test_index_numerical.json
    ├── remain_params_real.json         # Seen settings
    ├── in_dist_test_params_real.json   # In-distribution settings
    ├── out_dist_test_params_real.json  # Out-of-distribution settings
    ├── remain_params_numerical.json
    ├── in_dist_test_params_numerical.json
    └── out_dist_test_params_numerical.json
```

All configurations automatically read from `data/foil/hf_dataset` and apply `prefix_frames: 2000` to prevent data leakage.

---

## Usage Guide

### 1. Batch Training (Main Comparison Models)
Run all 10 baseline comparison models sequentially with automatic checkpoint archiving:

```bash
# Option A: Using the helper bash script
bash scripts/train_main.sh

# Option B: Direct Python command
python -m pod_sim2real.training.train --config-dir yaml_main/
```

### 2. Batch Training (Ablation Study)
Run all 6 ablation configurations:

```bash
# Option A: Using the helper bash script
bash scripts/train_ablation.sh

# Option B: Direct Python command
python -m pod_sim2real.training.train --config-dir yaml_ablation/
```

### 3. Running an Individual Config
```bash
python -m pod_sim2real.training.train --config yaml_main/09_triad-mno_config.yaml
```

### 4. One-Click Multi-Subset Evaluation
Evaluate all models archived in `best_checkpoints/` across `seen`, `in_dist`, `out_dist`, and `all` test splits:

```bash
# Option A: Using the helper bash script
bash scripts/evaluate.sh

# Option B: Direct Python command
python -m pod_sim2real.training.evaluate --checkpoints-dir best_checkpoints/
```

**Evaluation Outputs**:
- `evaluation_results_YYYYMMDD_HHMMSS.xlsx`:
  - **`Summary` Sheet**: Paper Table 1 format (Overall MSE, Rel-$L_2$, Vorticity MSE, Vorticity Rel-$L_2$).
  - **`Detailed_Subsets` Sheet**: Granular breakdown across `all`, `seen`, `in_dist`, and `out_dist`.
  - **`Per_Channel` Sheet**: Error metrics broken down by $u$ and $v$ velocity channels.
- `evaluation_results_YYYYMMDD_HHMMSS.json`: Machine-readable results.
- Formatted console summary table printed directly to stdout.

---

## Directory Structure

```text
POD-Sim2Real/
├── configs/                # Template and fallback configs
├── yaml_main/              # 10 official benchmark configs (00_unet to 09_triad-mno)
├── yaml_ablation/          # 6 ablation study configs
├── scripts/                # Bash & PowerShell launcher scripts
│   ├── train_main.sh
│   ├── train_ablation.sh
│   └── evaluate.sh
├── src/
│   └── pod_sim2real/       # Core package
│       ├── data/           # Arrow dataset streaming & windowing
│       ├── model/          # Pure PyTorch models (TriadMNO, FNO, AFNO, Transolver, etc.)
│       ├── training/       # Trainer, loss functions, batch runner, evaluator
│       └── visualization/  # Audit tools & video renderers
├── tests/                  # Pytest unit & regression tests
├── best_checkpoints/       # Automatically archived optimal weights & logs (gitignored)
├── artifacts/              # Intermediate training checkpoints and runs (gitignored)
├── requirements.txt        # PIP dependencies
├── pyproject.toml          # Build configuration
└── README.md
```

---

## Running Tests

Run the complete regression suite (22 unit tests covering models, datasets, loss functions, batch training, and evaluation):

```bash
pytest tests
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

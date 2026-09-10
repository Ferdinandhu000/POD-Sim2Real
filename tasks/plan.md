# Implementation Plan: Triad-MNO Architecture Modernization & SOTA Performance Optimization

## Overview
This plan modernizes the Triad-Coupled Modal Neural Operator (Triad-MNO) architecture in `POD-Sim2Real` to resolve the performance gap between MNO and strong baselines (FNO, AFNO, UNet), while strictly preserving and enhancing the paper's physical narrative (Hussain-Reynolds triple decomposition, Navier-Stokes triad coupling, continuous neural field super-resolution, and Sim-to-Real transferability).

## Architecture Decisions
- **Decision 1: Dual-Path Hybrid Micro Field (Basis-Guided + Continuous INR)**
  - *Rationale*: Pure coordinate MLP from a 32-dim latent vector discards $1\text{M}$ empirical POD basis parameters and creates an extreme information bottleneck. Combining linear micro-basis reconstruction with a continuous residual field guarantees a strong lower bound on accuracy while maintaining zero-shot super-resolution.
- **Decision 2: Fourier Features + FiLM Modulation in Continuous Neural Field Decoder**
  - *Rationale*: Standard coordinate MLPs suffer from severe spectral bias (low-frequency blur). Multi-frequency Fourier features ($\sin / \cos$) coupled with FiLM (Feature-wise Linear Modulation: $\gamma(z) \odot x + \beta(z)$) enable true spatial wave phase shifting and high-frequency vortex core resolution.
- **Decision 3: Soft Physics-Anchored Adaptation (Differential LR / Adapters) instead of Hard Freezing**
  - *Rationale*: Hard-freezing the macro backbone prevents the model from adapting to real PIV frequency/boundary layer shifts in 95% of flow energy. Differential learning rate ($0.1\times$) preserves simulation vortex physics while allowing necessary real-world domain alignment.
- **Decision 4: Loss Alignment & Curriculum Warmup**
  - *Rationale*: Triad-MNO was penalized by optimizing a composite MSE + Vorticity loss while early stopping and baselines only optimized pure MSE. We implement phased warmup and configurable loss weighting so MNO dominates both MSE and physical enstrophy metrics.

## Task List

### Phase 1: Foundations & Decoder Architecture (Vertical Slice 1)
- [ ] Task 1.1: Design and implement `FourierFeatureEncoding` and `FiLMLayer` in `models.py`.
- [ ] Task 1.2: Refactor `ContinuousNeuralFieldDecoder` to use Fourier features + FiLM modulation.
- [ ] Task 1.3: Implement Dual-Path Hybrid Micro Reconstruction in `TriadMNO`.
- [ ] Task 1.4: Write unit tests and parameter verification scripts (`tests/test_mno_v2.py`) to verify forward pass, super-resolution pass, and parameter count (~160k).

### Checkpoint 1: Foundations Verified
- [ ] All unit tests pass for MNO-v2 at native resolution ($64 \times 128$) and 2x super-resolution ($128 \times 256$).
- [ ] Output shapes and gradient flow verified.

### Phase 2: Sim-to-Real Adaptation & Loss Alignment (Vertical Slice 2)
- [ ] Task 2.1: Implement differential parameter grouping and learning rate scaling in `train_stage` (`trainer.py`).
- [ ] Task 2.2: Update `train.py` to support `soft_macro_adaptation` alongside legacy `freeze_macro`.
- [ ] Task 2.3: Update `losses.py` with curriculum vorticity weighting and balanced metric reporting.
- [ ] Task 2.4: Create new config files (`yaml_main/09_triad-mno_v2_config.yaml`) and smoke run verification.

### Checkpoint 2: Training Pipeline Verified
- [ ] 2-epoch pretrain + 2-epoch finetune runs cleanly on CPU/GPU without NaN or shape mismatch.
- [ ] Checkpoint saving, logging, and evaluation metrics function as expected.

### Phase 3: Paper Narrative & Evaluation Synchronization
- [ ] Task 3.1: Update theoretical formulation and equations in `Template.tex` to reflect the dual-path hybrid reconstruction and Fourier-FiLM neural field decoder.
- [ ] Task 3.2: Prepare updated ablation table rows comparing MNO-v1 vs MNO-v2.

## Risks and Mitigations
| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| FiLM decoder increases memory footprint during training | Medium | Keep latent dim at 64 and hidden dim at 64/128; use fused operations and chunked coordinate querying if needed. |
| Unfreezing macro backbone causes catastrophic forgetting of simulation vortex frequency | Medium | Use differential learning rate ($0.1 \times$ or $0.05 \times$) to keep macro modes closely anchored to Navier-Stokes kinematics. |
| Zero-shot spatial super-resolution API breaks | Low | Implement clean fallback: bicubic interpolation for linear macro/micro bases + continuous coordinate grid for FiLM decoder. |

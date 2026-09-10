# Tasks Todo: Triad-MNO Architecture Modernization

- [ ] **Phase 1: Foundations & Decoder Modernization**
  - [ ] 1.1 Add `FourierFeatureEncoding` and `FiLMLayer` to `src/pod_sim2real/model/models.py`
  - [ ] 1.2 Upgrade `ContinuousNeuralFieldDecoder` to FiLM-modulated Fourier architecture
  - [ ] 1.3 Implement Dual-Path Hybrid Micro Reconstruction in `TriadMNO`
  - [ ] 1.4 Write and run unit tests `tests/test_mno_v2.py`
- [ ] **Checkpoint 1: Architecture & Forward Pass Verification**
- [ ] **Phase 2: Sim-to-Real Adaptation & Loss Alignment**
  - [ ] 2.1 Add differential learning rates in `src/pod_sim2real/training/trainer.py`
  - [ ] 2.2 Wire soft adaptation into `src/pod_sim2real/training/train.py`
  - [ ] 2.3 Implement curriculum vorticity weighting in `src/pod_sim2real/training/losses.py`
  - [ ] 2.4 Create `yaml_main/09_triad-mno_v2_config.yaml`
- [ ] **Checkpoint 2: Smoke Train & Pipeline Verification**
- [ ] **Phase 3: Paper Story & Ablation Table**
  - [ ] 3.1 Update mathematical formulas in `Template.tex`
  - [ ] 3.2 Add ablation rows in paper draft

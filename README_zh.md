# POD-Sim2Real

[English](README.md) | [中文说明](README_zh.md)

用于 RealPDEBench NACA0025 水翼数据的 Sim-to-Real 流场预测实验，包含 Triad-MNO 主模型及标准神经算子对比基线。

---

## 1. 环境配置

```bash
# 1. 创建并激活 conda 虚拟环境
conda create -n realpde python=3.10 -y
conda activate realpde

# 2. 根据服务器显卡驱动安装 PyTorch (以 CUDA 11.8 为例)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装依赖并以可编辑模式注册当前包
pip install -r requirements.txt
pip install -e .
```

---

## 2. 数据准备

将下载好的官方数据集放置在 `data/foil/` 目录下：

```text
data/
└── foil/
    ├── hf_dataset/
    │   ├── real/                       # 包含 dataset_info.json, state.json, *.arrow
    │   ├── numerical/                  # 包含 dataset_info.json, state.json, *.arrow
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

> **说明**：程序默认统一固定截断使用前 2000 帧（`prefix_frames: 2000`）进行训练、验证与测试，严格杜绝时间泄露，无需手动预处理。

---

## 3. 模型训练

### 方式一：一键批量训练（推荐）
```bash
# 1. 顺序训练 yaml_main/ 下的所有主要对比模型（10 个模型）
bash scripts/train_main.sh
# 或执行: python -m pod_sim2real.training.train --config-dir yaml_main/

# 2. 顺序训练 yaml_ablation/ 下的所有消融模型（6 个消融实验）
bash scripts/train_ablation.sh
# 或执行: python -m pod_sim2real.training.train --config-dir yaml_ablation/
```

### 方式二：单独训练单个模型
```bash
python -m pod_sim2real.training.train --config yaml_main/09_triad-mno_config.yaml
```

---

## 4. 权重与日志保存路径

- **训练过程产物**：每个模型保存在独立的 `artifacts/runs/<yaml名称>/` 目录下（如 `artifacts/runs/09_triad-mno_config/`），互不覆盖。
- **最佳检查点自动归档**：触发早停（patience=10）后，程序会自动定位验证集指标最优的一轮（例如第 53 轮早停，最佳轮次为 43 轮），并将完整要素归档至 `best_checkpoints/<yaml名称>/`：
  - `best.pt`：Real 微调最佳模型权重
  - `best_sim.pt`：Sim 预训练最佳模型权重
  - `train.log`：完整训练日志
  - `config.yaml`：该实验完整配置
  - `pod_u.npz`, `pod_v.npz`：自包含的 POD 正交基底
  - `info.txt`：明确记录 `best_epoch` 及各项验证集误差指标

---

## 5. 一键评测与生成 Excel

训练完成后，对 `best_checkpoints/` 下所有模型执行批量评测：

```bash
bash scripts/evaluate.sh
# 或执行: python -m pod_sim2real.training.evaluate --checkpoints-dir best_checkpoints/
```

评测会自动测试 `seen`、`in_dist`、`out_dist` 及 `all` 四个官方划分，并生成：
- `evaluation_results_YYYYMMDD_HHMMSS.xlsx`：
  - `Summary`：对齐论文 Table 1 格式的汇总表（Overall MSE、Rel-L2、Vorticity MSE、Vorticity Rel-L2）；
  - `Detailed_Subsets`：各子集细分数据；
  - `Per_Channel`：u/v 独立速度分量误差指标。
- `evaluation_results_YYYYMMDD_HHMMSS.json`：同名原始 JSON。
- 终端同步打印对比表格。

---

## 6. 自动化测试

运行单元测试确认环境正常：
```bash
pytest tests
```

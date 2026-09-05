# POD-Sim2Real：基于 Triad-MNO 与神经算子的流场 Sim-to-Real 预测基准框架

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**English**](README.md) | [**简体中文**](README_zh.md)

本项目为 **Triad-MNO**（结合连续空间残差神经场的三元组耦合模态神经算子）的官方开源实现，并在 **RealPDEBench NACA0025 水翼** 数据集上构建了一套完整的 Sim-to-Real 流场预测实验评估体系。

项目支持数值仿真数据预训练（Sim Pre-training）、实验 PIV 数据微调（Real Fine-tuning）、自动化早停与最佳权重归档、以及一键多子集评估并导出符合顶会发表标准的 Excel 数据报表。

---

## 核心特性与创新

1. **Triad-MNO 核心架构**：
   - **Hussain-Reynolds 三重物理分解**：$\mathbf{u}(\mathbf{x}, t) = \overline{\mathbf{U}}(\mathbf{x}) + \widetilde{\mathbf{u}}(\mathbf{x}, t) + \mathbf{u}''(\mathbf{x}, t)$，解耦时间均值基流、宏观相干涡街结构（模态 1–32）与细粒度湍流脉动（模态 33–64）。
   - **连续坐标神经场解码器（Continuous Neural Field Decoder）**：利用隐式坐标表征 $(x, y) \in [-1, 1]^2 \to \Delta \mathbf{u}(\mathbf{x}, t)$ 查询高阶微模态，彻底消除传统 POD 的模态淹没问题，并具备**零样本空间超分辨率**泛化能力。
   - **跨尺度三元组注意力（Cross-Scale Triad Attention）**：参数化 Navier-Stokes 方程二次对流三元组张量作用（$Q_{ijk} a_j a_k$），捕捉宏观涡与微观湍流间的能量跨尺度传递。
   - **涡度/拟涡能物理正则化（Vorticity / Enstrophy Loss）**：采用中心有限差分涡度损失作为高通滤波器，抑制高频拟涡能过度耗散。
   - **宏观基底冻结策略（`freeze_macro: true`）**：在真实实验数据微调阶段冻结宏观主干，锁死普适的脱落涡涡频，仅自适应微调近壁面边界层修正。

2. **严谨的 2000 帧截断协议（`prefix_frames: 2000`）**：
   - 彻底废除比例切分，对训练、验证、测试所有轨迹硬性截断前 2000 帧，坚决杜绝时间维度上的数据穿越与泄漏。

3. **批量配置执行与模型输出完全隔离**：
   - 终端使用 `--config-dir` 即可一键顺序执行目录内的全部 YAML 配置。
   - 每个实验生成以 YAML 文件名命名的独立输出目录（如 `artifacts/runs/00_unet_config/`），保证各模型互不覆盖。

4. **早停最佳权重自动归档（`best_checkpoints/`）**：
   - 当早停（patience=10）触发后，系统自动定位真正的验证集最优轮次（例如第 53 轮早停，最佳轮次为 43 轮）。
   - 自动将 `best.pt`、`best_sim.pt`、`train.log`、`config.yaml`、`results.json`、`split_manifest.json`、POD 基文件（`pod_u.npz`, `pod_v.npz`）归档，并生成 `info.txt` 记录最优指标。

5. **一键多子集评估与 Excel 报表自动生成**：
   - 一键评测 `best_checkpoints/` 下所有模型，覆盖官方划分的 `seen`（同工况测试集）、`in_dist`（分布内未见工况）、`out_dist`（分布外未见工况）及 `all`（总体加权）。
   - 自动输出带时间戳的 `evaluation_results_YYYYMMDD_HHMMSS.xlsx`（包含 `Summary` 对齐论文 Table 1、`Detailed_Subsets`、`Per_Channel` 速度分量三张 Sheet）。

---

## 实验体系模型一览

### 1. 主对比实验配置（`yaml_main/`）
| 配置名称 | 对应模型 | 架构类别 | 说明 |
|:---|:---|:---|:---|
| `00_unet_config.yaml` | `unet` | 空间网格直接建模 | 经典 2D 空间 U-Net 基线 |
| `01_fno_config.yaml` | `fno` | 空间网格直接建模 | 2D 傅里叶神经算子（Fourier Neural Operator） |
| `02_afno_config.yaml` | `afno` | 空间网格直接建模 | 2D 自适应傅里叶神经算子（Adaptive FNO） |
| `03_itransolver_config.yaml` | `itransolver` | 空间网格直接建模 | 空间网格物理注意力 Transolver |
| `04_pod-unet_config.yaml` | `pod-unet` | 模态降阶（ROM） | 32 阶 POD 正交基 + 1D U-Net 模态系数动力学 |
| `05_pod-fno_config.yaml` | `pod-fno` | 模态降阶（ROM） | 32 阶 POD 正交基 + 1D FNO 模态系数动力学 |
| `06_pod-afno_config.yaml` | `pod-afno` | 模态降阶（ROM） | 32 阶 POD 正交基 + 1D AFNO 模态系数动力学 |
| `07_pod-itransformer_config.yaml` | `pod-itransformer` | 模态降阶（ROM） | 32 阶 POD 正交基 + Inverted Transformer |
| `08_pod-transolver_config.yaml` | `pod-transolver` | 模态降阶（ROM） | 32 阶 POD 正交基 + 1D Transolver |
| `09_triad-mno_config.yaml` | `triad-mno` | 多尺度混合表征 | **Triad-MNO（完整提出模型）** |

### 2. 系统消融实验配置（`yaml_ablation/`）
| 配置名称 | 测试变体 | 说明 |
|:---|:---|:---|
| `00_triad-mno_full_config.yaml` | 完整 Triad-MNO | 完整模型对比参考基准 |
| `01_triad-mno_no_attn_config.yaml` | 去除 Triad 注意力 | 关闭宏观与微观模态之间的交叉注意力耦合 |
| `02_triad-mno_linear64_config.yaml` | 去除连续神经场 | 替换为传统线性 64 模态 SVD 投影重建 |
| `03_triad-mno_no_vort_config.yaml` | 去除涡度损失 | 仅使用纯速度场 MSE 损失进行训练 |
| `04_triad-mno_no_freeze_config.yaml` | 去除宏观冻结 | 真实实验微调时不冻结宏观主干参数 |
| `05_triad-mno_macro_only_config.yaml` | 纯宏观基线 | 仅保留 32 阶宏观算子（$r_{\mathrm{macro}}=32, r_{\mathrm{micro}}=0$） |

---

## 环境安装与服务器配置

### 步骤 1：创建 Conda 虚拟环境
```bash
conda create -n realpde python=3.10 -y
conda activate realpde
```

### 步骤 2：安装适配服务器 CUDA 版本的 PyTorch
推荐优先通过官方源安装与显卡驱动匹配的 PyTorch：

```bash
# 若服务器为 CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 若服务器为 CUDA 12.1 / 12.4:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 步骤 3：安装项目依赖与注册包
```bash
# 安装数据处理、指标计算与 Excel 生成工具
pip install -r requirements.txt

# 以可编辑模式注册当前项目
pip install -e .
```

---

## 数据集准备规范

从 Hugging Face 下载官方 **RealPDEBench Foil** 数据集，确保完整目录结构存放在 `data/foil` 下：

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
    ├── remain_params_real.json         # Seen 工况
    ├── in_dist_test_params_real.json   # In-distribution 分布内工况
    ├── out_dist_test_params_real.json  # Out-of-distribution 分布外工况
    ├── remain_params_numerical.json
    ├── in_dist_test_params_numerical.json
    └── out_dist_test_params_numerical.json
```

---

## 运行与评估流程

### 1. 一键批量运行主对比实验
自动顺序训练 `yaml_main/` 下的 10 个基线模型，并在触发早停后将最优检查点自动归档至 `best_checkpoints/`：

```bash
# 方式 A：使用 Linux 便捷 Bash 脚本（推荐）
bash scripts/train_main.sh

# 方式 B：通过 Python 模块直接调用
python -m pod_sim2real.training.train --config-dir yaml_main/
```

### 2. 一键批量运行消融实验
顺序训练 6 个消融变体模型：

```bash
# 方式 A：使用 Linux 便捷 Bash 脚本（推荐）
bash scripts/train_ablation.sh

# 方式 B：通过 Python 模块直接调用
python -m pod_sim2real.training.train --config-dir yaml_ablation/
```

### 3. 运行单个模型配置
```bash
python -m pod_sim2real.training.train --config yaml_main/09_triad-mno_config.yaml
```

### 4. 一键多子集评估与 Excel 报表导出
训练完成后，对 `best_checkpoints/` 目录下一键运行全量子集评测：

```bash
# 方式 A：使用 Linux 便捷 Bash 脚本（推荐）
bash scripts/evaluate.sh

# 方式 B：通过 Python 模块直接调用
python -m pod_sim2real.training.evaluate --checkpoints-dir best_checkpoints/
```

**生成的评测产物**：
- `evaluation_results_YYYYMMDD_HHMMSS.xlsx`：
  - **`Summary` Sheet**：完全对齐论文 **Table 1** 格式（Overall MSE、Rel-$L_2$、Vorticity MSE、Vorticity Rel-$L_2$）；
  - **`Detailed_Subsets` Sheet**：详细展开各模型在 `all`、`seen`、`in_dist`、`out_dist` 下的表现；
  - **`Per_Channel` Sheet**：独立展开流场水平分量 $u$ 与垂直分量 $v$ 的误差指标。
- `evaluation_results_YYYYMMDD_HHMMSS.json`：便于脚本程序化读取。
- 终端同步打印美观的 ASCII 对比表格。

---

## 代码目录结构

```text
POD-Sim2Real/
├── configs/                # 历史模板与基础配置
├── yaml_main/              # 10 个主要对比实验配置文件 (00_unet 至 09_triad-mno)
├── yaml_ablation/          # 6 个系统消融实验配置文件
├── scripts/                # Linux Bash 与 Windows 启动脚本
│   ├── train_main.sh       # 批量训练主对比模型
│   ├── train_ablation.sh   # 批量训练消融模型
│   └── evaluate.sh         # 一键评估与 Excel 导出
├── src/
│   └── pod_sim2real/       # 核心实现包
│       ├── data/           # Arrow 数据流式读取与样本切片
│       ├── model/          # 纯原生 PyTorch 模型库 (TriadMNO, FNO, AFNO, Transolver 等)
│       ├── training/       # 训练器、涡度物理损失、批处理执行器、多子集评估器
│       └── visualization/  # 数据审计与对比视频生成
├── tests/                  # Pytest 自动化测试套件
├── best_checkpoints/       # 自动归档的最佳模型权重与日志 (已加入 .gitignore)
├── artifacts/              # 训练中间过程检查点与中间文件 (已加入 .gitignore)
├── requirements.txt        # 依赖列表
├── pyproject.toml          # 包构建与命令注册
├── README.md               # 英文说明文档
└── README_zh.md            # 中文说明文档
```

---

## 自动化测试

运行项目完整的回归测试套件（覆盖所有模型、数据流、损失函数、批训练与评测逻辑共 22 个测试用例）：

```bash
pytest tests
```

---

## 开源协议

本项目遵循 [MIT License](LICENSE) 开源协议。

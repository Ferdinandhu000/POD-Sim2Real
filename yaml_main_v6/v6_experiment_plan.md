# v6 实验矩阵与架构升级报告 (v6 Experiment Plan)

本报告详细记录 `yaml_main_v6/` 目录下全新研发的 **POD-ResUNet3D（模态残差时空 UNet 算子）** 核心架构原理、对齐官方基线与冲击 SOTA 设计、双流耦合机制以及官方评测规则统一规范。

---

## 一、 核心攻关背景与官方基线对齐

在 RealPDEBench 官方论文 Table 1（Foil 水翼场景）中：
- **U-Net** 是所有神经算子基线中表现最顶尖的 SOTA 模型：
  - **参数量**: 23.0 M
  - **RMSE**: **0.0094**（远超 FNO 的 0.0120、CNO 的 0.0114、DPOT-L 的 0.0109）
  - **Rel $L_2$**: **0.0145** (1.45%)
  - **fRMSE**: **0.0008**
- **本项目的 UNet3D 传承验证**：
  - `src/pod_sim2real/model/realpdebench/unet.py` 100% 完整复刻自 RealPDEBench 官方的 3D 视频扩散 UNet 算子；
  - 包含 3D 时空残差卷积块（`ResnetBlock` + `GroupNorm(8)`）、空间线性自注意力（`SpatialLinearAttention`）、带旋转位置编码（`RotaryEmbedding`）与相对位置偏置（`RelativePositionBias`）的时序分解注意力（`temporal_attn`）；
  - 当配置为 `dim=64, dim_mults=(1, 2, 4), channels=2` 时，模型参数量精确为 **22.96 M**（论文四舍五入为 **23.0 M**），实现与官方算子结构的严密数学对齐。

---

## 二、 模型设计逻辑升级：与 v5 POD-Transformer 保持一致

在前期设计中，POD-UNet 曾采用通道拼接（`channel_concat`），将历史时刻 $t$ 与粗场未来时刻 $t+20$ 拼接为 4 通道，导致 3D 卷积核在时序维度同时混合了跨越 20 步的不同物理流态。

**v6 升级后的核心逻辑**：
1. **统一双流时空耦合范式（`temporal_concat`，默认）**：
   - 将观测历史 $\mathbf{u}_{\text{in}}$（20 步）与宏观粗场 $\widehat{\mathbf{u}}_{\text{pod}}$（20 步）沿时间轴串接为连续的 **40 帧流体时空块** $[B, 40, H, W, 2]$；
   - 保证通道数仍为物理速度场 2 通道（$u, v$），使 3D 卷积与时序注意力在完整、连续的流体物理时间线上提取时空特征；
   - 经由 3D-UNet 编码-解码后，通过时间投影头 `time_project(40 -> 20)` 无缝映射为未来 20 步的高频残差场 $\Delta \mathbf{u}$；
   - 最终输出为：$\mathbf{u}_{\text{final}} = \widehat{\mathbf{u}}_{\text{pod}} + \Delta \mathbf{u}$。
2. **多模式消融支持**：
   - `temporal_concat`（默认，旗舰模式）：40 步流体块输入 + 40->20 时序投影；
   - `future_refine`（细化消融）：UNet3D 专门作为粗场细节超分网络，输入 $\widehat{\mathbf{u}}_{\text{pod}}$（20 步）输出 $\Delta \mathbf{u}$（20 步）；
   - `channel_concat`（历史消融）：4 通道输入，对比时序串接与通道堆叠的物理表征差异。
3. **显存与 Batch Size 调优**：
   - 3D-UNet 激活值显存较大（单样本反向传播约需 5.1 GB）；
   - 针对 80GB/84GB GPU（如 RTX 6000D / A100），将批大小严格设为 **`batch_size: 8`**（反向显存约 41 GB），彻底杜绝 CUDA OOM 风险。

---

## 三、 评测规则统一：对齐 RealPDEBench 官方 Rel-$L_2$

### 1. 差异根源诊断
- **现象**：在 Foil 场景复现实验中，FNO3D 跑出的 RMSE 为 **0.0118023**，与论文 Table 1 的 **0.0120** 高度吻合（误差 < 1.6%）；但以前代码输出的 Rel-$L_2$ 为 **4.52%**，论文为 **2.06%**。
- **原因**：
  - **旧代码计算规则（全局范数比值）**：
    $$\text{Rel-}L_2^{\text{old}} = \frac{\sqrt{\sum_{i=1}^N \|\mathbf{y}_i - \widehat{\mathbf{y}}_i\|_2^2}}{\sqrt{\sum_{i=1}^N \|\mathbf{y}_i\|_2^2}} = \frac{\text{RMSE}}{\text{RMS}(\mathbf{y}_{\text{global}})}$$
    在测试集全部 5140 个窗口中，由于包含大迎角失速区分离流（平均速度低，RMS 仅 ~0.21），全场分母均方根 $\text{RMS}(\mathbf{y}_{\text{global}}) \approx 0.261$。$0.0118 / 0.261 = \mathbf{4.52\%}$。
  - **RealPDEBench 官方规则（样本级逐窗口均值比值）**：
    在 `realpdebench/utils/metrics.py` 第 55-57 行中，官方标准实现为：
    ```python
    err_l2 = torch.norm(pred.reshape(b, -1) - target.reshape(b, -1), dim=1)
    norm = torch.norm(target.reshape(b, -1), dim=1)
    rel_l2_error = torch.mean(err_l2 / norm)
    ```
    即对每个测试样本（20 帧窗口）分别除以该样本自身的范数，再取全局均值：
    $$\text{Rel-}L_2^{\text{official}} = \frac{1}{N} \sum_{i=1}^N \frac{\|\mathbf{y}_i - \widehat{\mathbf{y}}_i\|_2}{\|\mathbf{y}_i\|_2}$$

### 2. 统一后的代码与接口
- `src/pod_sim2real/training/losses.py`：新增 `relative_l2_per_sample` 与 `mean_relative_l2`，并在 `compute_metrics` 中默认返回样本级相对误差；
- `src/pod_sim2real/training/trainer.py`：`evaluate_model` 严格采用样本级累加除以样本总数计算 `rel_l2`、`u_rel_l2`、`v_rel_l2`、`vorticity_rel_l2`；
- 同时在返回字典中保留 `global_rel_l2` 字段，确保与历史实验日志完全向前兼容。

---

## 四、 v6 实验矩阵规划详表

| 序号 | 配置文件名 | 核心模型 | 架构类别 | 关键配置与参数 | 批大小 / 学习率 | 核心目的与预期 |
| :---: | :--- | :---: | :---: | :--- | :---: | :--- |
| **00** | `6_00_pod_res_unet3d_rank96.yaml` | **POD-ResUNet3D** | **冲击 SOTA 旗舰** | POD Rank 96, UNet3D (23M, 40->20 双流时序), $v$-weight=3.0, $\mathcal{L}_\omega=0.1$, $\mathcal{L}_{\text{pod}}=0.5$ | 8 / 2e-4 | **对齐并超越最强基线 UNet (0.0145 / 0.0094)**，冲击全场最低误差新 SOTA。 |
| **01** | `6_01_pod_res_unet3d_rank64.yaml` | **POD-ResUNet3D** | 模态阶数消融 | POD Rank 64, 其余配置同 00 | 8 / 2e-4 | 评估当模态数从 96 降低至 64 时，UNet3D 残差细化模块的鲁棒性。 |
| **02** | `6_02_pod_res_unet3d_pure_mse.yaml` | **POD-ResUNet3D** | 复合损失消融 | POD Rank 96, 纯 MSE 损失（无 $v$-weight，无涡度损失） | 8 / 2e-4 | 剥离复合物理损失，验证双流 UNet 时空架构本身的纯物理建模增益。 |
| **03** | `6_03_pod_res_unet3d_future_refine.yaml` | **POD-ResUNet3D** | 耦合机制消融 | POD Rank 96, `future_refine` 模式（UNet 仅输入 20 步粗场做细节超分） | 8 / 2e-4 | 对比 40 步时序拼接与纯粗场细化，定量验证双流历史注入的必要性。 |

---

## 五、 调度与执行指南

```bash
# 1. 单卡运行全部 v6 实验：
CUDA_VISIBLE_DEVICES=0 bash scripts/train_main_v6.sh

# 2. 多卡并行调度（例如 4 卡）：
bash scripts/train_main_v6.sh --gpus 0,1,2,3

# 3. 单独测试旗舰 SOTA 模型：
bash scripts/train_main_v6.sh --config yaml_main_v6/6_00_pod_res_unet3d_rank96.yaml --gpus 0
```

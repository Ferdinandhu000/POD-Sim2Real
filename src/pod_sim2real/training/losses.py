from __future__ import annotations
import math
import torch
from torch.nn import functional as F

def relative_tke_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative kinetic energy loss."""
    pred_ke = 0.5 * (pred[:, :, 0] ** 2 + pred[:, :, 1] ** 2).mean(dim=(-2, -1))
    target_ke = 0.5 * (target[:, :, 0] ** 2 + target[:, :, 1] ** 2).mean(dim=(-2, -1))
    diff = (pred_ke - target_ke).abs()
    denom = target_ke.clamp(min=1e-5)
    return (diff / denom).mean()

def compute_vorticity(flow: torch.Tensor) -> torch.Tensor:
    """Compute 2D vorticity field omega = dv/dx - du/dy.
    flow: [..., 2, H, W]
    """
    orig_shape = flow.shape[:-3]
    h, w = flow.shape[-2], flow.shape[-1]
    flat = flow.reshape(-1, 2, h, w)
    u = flat[:, 0:1]
    v = flat[:, 1:2]

    u_pad = F.pad(u, (0, 0, 1, 1), mode="replicate")
    v_pad = F.pad(v, (1, 1, 0, 0), mode="replicate")

    du_dy = (u_pad[:, :, 2:, :] - u_pad[:, :, :-2, :]) * 0.5
    dv_dx = (v_pad[:, :, :, 2:] - v_pad[:, :, :, :-2]) * 0.5

    omega = dv_dx - du_dy
    return omega.reshape(*orig_shape, h, w)

def vorticity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Enstrophy / vorticity MSE loss between predicted and target velocity fields."""
    omega_pred = compute_vorticity(pred)
    omega_target = compute_vorticity(target)
    return F.mse_loss(omega_pred, omega_target)


def relative_l2_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the official RealPDEBench relative L2 error for each sample.

    RealPDEBench computes ``||pred-target||_2 / ||target||_2`` after flattening
    each sample, then averages those ratios across samples.  A zero target is
    treated as a zero-error sample when its prediction is also zero; otherwise
    its ratio is finite and well-defined using a unit denominator.
    """
    if pred.shape[0] != target.shape[0]:
        raise ValueError("pred and target must have the same batch dimension")
    pred_flat = pred.float().reshape(pred.shape[0], -1)
    target_flat = target.float().reshape(target.shape[0], -1)
    err = torch.linalg.vector_norm(pred_flat - target_flat, dim=1)
    norm = torch.linalg.vector_norm(target_flat, dim=1)
    # Official data has non-zero norms. Keep degenerate synthetic/constant
    # fields finite so evaluation remains usable instead of returning NaN.
    return torch.where(norm > 0, err / norm, torch.where(err == 0, err, err / 1.0))


def mean_relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean official-style per-sample relative L2 error."""
    return relative_l2_per_sample(pred, target).mean()

def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    kind: str = "",
    field_weight: float = 0.0,
    tke_weight: float = 0.05,
    use_tke: bool | None = None,
    vorticity_weight: float = 0.1,
    use_vorticity: bool | None = None,
    return_scalars: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
    """Compute training loss.
    
    Base field loss is standard MSE.
    Optional physics auxiliary losses:
    - TKE (turbulent kinetic energy) loss
    - Vorticity / Enstrophy gradient loss (forces resolution of small-scale vortex cores)
    """
    if isinstance(pred, (tuple, list)):
        pred_main = pred[0]
        pred_aux = pred[1] if len(pred) > 1 else None
    else:
        pred_main = pred
        pred_aux = None

    v_weight = float(kwargs.get("v_weight", 1.0))
    if pred_main.ndim >= 4 and pred_main.shape[2] == 2 and v_weight != 1.0:
        mse_u = F.mse_loss(pred_main[:, :, 0], target[:, :, 0])
        mse_v = F.mse_loss(pred_main[:, :, 1], target[:, :, 1])
        mse = (mse_u + v_weight * mse_v) / (1.0 + v_weight) * 2.0
    else:
        mse = F.mse_loss(pred_main, target)

    pod_weight = float(kwargs.get("pod_weight", 0.0))
    if pred_aux is not None and pod_weight > 0.0:
        mse = mse + pod_weight * F.mse_loss(pred_aux, target)

    enable_tke = use_tke if use_tke is not None else (kind.startswith("pod-") and tke_weight > 0)
    tke = relative_tke_loss(pred_main, target) if enable_tke else pred_main.new_zeros(())

    enable_vort = use_vorticity if use_vorticity is not None else ("triad" in kind or "res" in kind)
    vort = vorticity_loss(pred_main, target) if enable_vort and pred_main.ndim >= 4 and pred_main.shape[2] == 2 else pred_main.new_zeros(())

    total_loss = mse
    if enable_tke:
        total_loss = total_loss + tke_weight * tke
    if enable_vort:
        total_loss = total_loss + vorticity_weight * vort

    if return_scalars:
        return total_loss, {
            "loss": float(total_loss.detach()),
            "mse": float(mse.detach()),
            "tke": float(tke.detach()),
            "vorticity": float(vort.detach()),
        }
    # Keep diagnostics on the current device during training. Converting these
    # values to Python floats here would synchronize the CUDA stream every step.
    return total_loss, {
        "loss": total_loss.detach(),
        "mse": mse.detach(),
        "tke": tke.detach(),
        "vorticity": vort.detach(),
    }

@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute comprehensive evaluation metrics for paper tables:
    - MSE: overall mean squared error
    - RMSE: root mean squared error
    - MAE: mean absolute error
    - Relative L2: relative Frobenius / L2 error ||pred - target||_2 / ||target||_2
    - u_mse, v_mse, u_rel_l2, v_rel_l2, etc.: per-channel breakdowns
    """
    diff = pred - target
    mse = float(diff.square().mean().item())
    rmse = float(math.sqrt(max(mse, 0.0)))
    mae = float(diff.abs().mean().item())

    rel_l2 = float(mean_relative_l2(pred, target).item())

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "rel_l2": rel_l2,
    }
    if pred.ndim >= 3 and pred.shape[2] == 2:
        u_diff, v_diff = diff[:, :, 0], diff[:, :, 1]
        u_tgt, v_tgt = target[:, :, 0], target[:, :, 1]
        u_mse = float(u_diff.square().mean().item())
        v_mse = float(v_diff.square().mean().item())
        metrics["u_mse"] = u_mse
        metrics["v_mse"] = v_mse
        metrics["u_rmse"] = float(math.sqrt(max(u_mse, 0.0)))
        metrics["v_rmse"] = float(math.sqrt(max(v_mse, 0.0)))
        metrics["u_mae"] = float(u_diff.abs().mean().item())
        metrics["v_mae"] = float(v_diff.abs().mean().item())
        metrics["u_rel_l2"] = float(mean_relative_l2(pred[:, :, 0], target[:, :, 0]).item())
        metrics["v_rel_l2"] = float(mean_relative_l2(pred[:, :, 1], target[:, :, 1]).item())
        omega_pred = compute_vorticity(pred)
        omega_tgt = compute_vorticity(target)
        omega_diff = omega_pred - omega_tgt
        metrics["vorticity_mse"] = float(omega_diff.square().mean().item())
        metrics["vorticity_rel_l2"] = float(mean_relative_l2(omega_pred, omega_tgt).item())
    return metrics

from __future__ import annotations
import copy
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .losses import compute_loss
from .logging_utils import log_metrics


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for a, b in zip(self.model.parameters(), model.parameters()):
            a.mul_(self.decay).add_(b, alpha=1 - self.decay)


def _save(path, payload):
    torch.save(payload, path)


def train_stage(model, train_ds, val_ds, stage_dir, config, stage, device, initial=None):
    stage_dir = Path(stage_dir)
    (stage_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger = config["logger"]
    if initial is not None:
        model.load_state_dict(initial, strict=False)
    model.to(device)
    loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=config["lr"], weight_decay=config["weight_decay"])
    ema = EMA(model, config["ema_decay"])
    best = float("inf")
    best_epoch = 0
    bad = 0
    history = []
    start_epoch = 1

    use_amp = bool(config.get("use_amp", True)) and (device.type == "cuda")
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def _autocast():
        if hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", enabled=use_amp)
        return torch.cuda.amp.autocast(enabled=use_amp)

    resume = config.get("resume")
    if resume:
        resume_path = Path(resume)
        if resume_path.exists():
            payload = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(payload.get("raw_model_state", payload["model_state"]), strict=False)
            ema.model.load_state_dict(payload.get("model_state", payload["raw_model_state"]), strict=False)
            if payload.get("optimizer_state"):
                opt.load_state_dict(payload["optimizer_state"])
            best = float(payload.get("best_val_loss", best))
            best_epoch = int(payload.get("best_epoch", payload.get("epoch", 0)))
            start_epoch = int(payload.get("epoch", 0)) + 1
            logger.info("resumed %s from %s at epoch %d", stage, resume_path, start_epoch)
        else:
            logger.warning("resume checkpoint does not exist: %s", resume_path)

    epochs = config["epochs"]
    patience = int(config.get("patience", 10))
    for epoch in range(start_epoch, epochs + 1):
        start = time.perf_counter()
        model.train()
        totals = {"loss": 0.0, "mse": 0.0, "tke": 0.0}
        bar = tqdm(loader, desc=f"{stage} Epoch {epoch:03d}/{epochs:03d} train", leave=False)
        for x, y, _ in bar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with _autocast():
                pred = model(x)
                loss, parts = compute_loss(
                    pred,
                    y,
                    kind=config["model"],
                    field_weight=config.get("field_weight", 0.0),
                    tke_weight=config.get("tke_weight", 0.05),
                    use_tke=config.get("use_tke", None),
                    vorticity_weight=config.get("vorticity_weight", 0.1),
                    use_vorticity=config.get("use_vorticity", None),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if config.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            totals["loss"] += loss.item()
            totals["mse"] += float(parts["mse"])
            totals["tke"] += float(parts["tke"])
            bar.set_postfix(loss=f"{loss.item():.4g}", mse=f"{float(parts['mse']):.4g}")

        val_sum = 0.0
        n = 0
        with torch.inference_mode():
            for x, y, _ in tqdm(val_loader, desc=f"{stage} Epoch {epoch:03d}/{epochs:03d} val", leave=False):
                x, y = x.to(device), y.to(device)
                with _autocast():
                    pred = ema.model(x)
                val_sum += torch.nn.functional.mse_loss(pred.float(), y.float()).item() * len(x)
                n += len(x)
        val = val_sum / max(n, 1)
        steps = max(len(loader), 1)
        metrics = {
            "stage": stage,
            "epoch": epoch,
            "train_loss": totals["loss"] / steps,
            "train_mse": totals["mse"] / steps,
            "tke_loss": totals["tke"] / steps,
            "val_loss": val,
            "lr": opt.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - start,
        }
        history.append(metrics)
        log_metrics(logger, metrics)
        logger.info(
            "%s epoch=%d val_loss=%.6g lr=%.3g time=%.2fs",
            stage,
            epoch,
            val,
            metrics["lr"],
            metrics["epoch_seconds"],
        )
        payload = {
            "model_state": ema.model.state_dict(),
            "raw_model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scheduler_state": None,
            "ema_state": ema.model.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": min(best, val),
            "model": config["model"],
            "stage": stage,
            "config": {k: v for k, v in config.items() if k not in ("logger", "resume")},
            "random_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        }
        _save(stage_dir / "last.pt", payload)
        if config["save_epoch_checkpoints"]:
            _save(stage_dir / f"checkpoint_epoch_{epoch:03d}.pt", payload)
        if val < best:
            best = val
            best_epoch = epoch
            bad = 0
            _save(stage_dir / "best.pt", payload)
        else:
            bad += 1
        if bad >= patience:
            logger.info("early stopping triggered at epoch %d (patience=%d)", epoch, patience)
            break

    (stage_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    logger.info("%s complete: best_epoch=%d best_val_loss=%.6g", stage, best_epoch, best)
    return ema.model.state_dict(), best, best_epoch


def evaluate_model(model, dataset, device, batch_size=1, num_workers=0) -> dict[str, float]:
    """Evaluate comprehensive paper metrics on a dataset:
    Returns dict with:
    - mse
    - rmse
    - mae
    - rel_l2
    - u_mse, v_mse, u_rmse, v_rmse, u_mae, v_mae, u_rel_l2, v_rel_l2
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.to(device).eval()

    total_sq_err = 0.0
    total_abs_err = 0.0
    total_tgt_sq = 0.0
    total_elements = 0

    u_sq_err = 0.0
    u_abs_err = 0.0
    u_tgt_sq = 0.0
    u_elements = 0

    v_sq_err = 0.0
    v_abs_err = 0.0
    v_tgt_sq = 0.0
    v_elements = 0

    with torch.inference_mode():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            diff = pred.float() - y.float()

            total_sq_err += diff.square().sum().item()
            total_abs_err += diff.abs().sum().item()
            total_tgt_sq += y.float().square().sum().item()
            total_elements += y.numel()

            if y.ndim >= 3 and y.shape[2] == 2:
                u_d = diff[:, :, 0]
                v_d = diff[:, :, 1]
                u_y = y[:, :, 0].float()
                v_y = y[:, :, 1].float()

                u_sq_err += u_d.square().sum().item()
                u_abs_err += u_d.abs().sum().item()
                u_tgt_sq += u_y.square().sum().item()
                u_elements += u_y.numel()

                v_sq_err += v_d.square().sum().item()
                v_abs_err += v_d.abs().sum().item()
                v_tgt_sq += v_y.square().sum().item()
                v_elements += v_y.numel()

    mse = total_sq_err / max(total_elements, 1)
    rmse = math.sqrt(max(mse, 0.0))
    mae = total_abs_err / max(total_elements, 1)
    rel_l2 = math.sqrt(total_sq_err) / (math.sqrt(total_tgt_sq) + 1e-8)

    res = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "rel_l2": rel_l2,
    }
    if u_elements > 0:
        u_mse = u_sq_err / u_elements
        v_mse = v_sq_err / v_elements
        res.update({
            "u_mse": u_mse,
            "u_rmse": math.sqrt(max(u_mse, 0.0)),
            "u_mae": u_abs_err / u_elements,
            "u_rel_l2": math.sqrt(u_sq_err) / (math.sqrt(u_tgt_sq) + 1e-8),
            "v_mse": v_mse,
            "v_rmse": math.sqrt(max(v_mse, 0.0)),
            "v_mae": v_abs_err / v_elements,
            "v_rel_l2": math.sqrt(v_sq_err) / (math.sqrt(v_tgt_sq) + 1e-8),
        })
    return res


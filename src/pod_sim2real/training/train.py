from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..data import (
    GaussianNormalizer,
    IdentityNormalizer,
    OfficialArrowWindowDataset,
    PrecomputedTrajectoryDataset,
    build_normalizer,
    build_trajectory_prefix_entries,
    discover_trajectories,
    preprocess_domain,
)
from ..data.arrow_dataset import _load_index_trajectory_ids
from ..data.splits import split_settings
from ..model import (
    build_model,
    fit_joint_pod_bases_from_datasets,
    fit_pod_bases,
    fit_pod_bases_from_dataset,
)
from .logging_utils import make_logger
from .trainer import evaluate_model, train_stage

MODELS = (
    "unet",
    "fno",
    "afno",
    "itransolver",
    "pod-fno",
    "pod-afno",
    "pod-unet",
    "pod-itransformer",
    "pod-transformer",
    "pod-itransolver",
    "pod-transolver",
    "triad-mno",
    "triad-afno",
    "fno3d",
    "unet3d",
    "afno3d",
    "transformer3d",
    "transolver3d",
    "itransolver3d",
)



def _resolve_data_layout(root: Path, data: dict) -> tuple[Path, Path, Path | None]:
    """Resolve legacy smoke paths or the official foil Arrow layout."""
    configured_real = Path(str(data.get("real_dir", "data/data_real")))
    configured_sim = Path(str(data.get("sim_dir", "data/data_sim")))
    legacy_pairs = [(configured_real, configured_sim)]
    official_pairs = []
    for base in (root, root / "data"):
        official_pairs.append((base / "foil/hf_dataset/real", base / "foil/hf_dataset/numerical"))
    pairs = official_pairs + legacy_pairs if data.get("use_official_indices", False) else legacy_pairs
    for real_dir, sim_dir in pairs:
        if not real_dir.is_absolute():
            real_dir, sim_dir = root / real_dir, root / sim_dir
        if real_dir.exists() and sim_dir.exists() and list(real_dir.glob("*.arrow")) and list(sim_dir.glob("*.arrow")):
            index_root = real_dir.parent
            has_indices = all((index_root / f"{split}_index_{suffix}.json").exists() for split in ("train", "val", "test") for suffix in ("real", "numerical"))
            return real_dir, sim_dir, index_root if has_indices else None
    raise FileNotFoundError(f"could not find Arrow data under {root}")


def _build_dataset(
    arrow_dir,
    index_root,
    suffix,
    split,
    input_steps,
    output_steps,
    resolution,
    test_mode="all",
    prefix_frames=None,
    *,
    index_entries=None,
    tensor_dir=None,
    cache_trajectories=True,
    max_cache_trajectories=8,
    in_memory=False,
    mmap=True,
):
    if tensor_dir is not None and tensor_dir.exists() and list(tensor_dir.glob("*.pt")):
        idx_file = (index_root / f"{split}_index_{suffix}.json") if index_root and suffix else None
        return PrecomputedTrajectoryDataset(
            tensor_dir=tensor_dir,
            index_file=idx_file,
            index_entries=index_entries,
            input_steps=input_steps,
            output_steps=output_steps,
            resolution=resolution,
            test_mode=test_mode if split in {"val", "test"} else "all",
            metadata_root=index_root.parent if index_root else None,
            prefix_frames=prefix_frames,
            in_memory=in_memory,
            mmap=mmap,
        )
    mode = test_mode if split in {"val", "test"} else "all"
    if index_entries is not None:
        return OfficialArrowWindowDataset(
            arrow_dir,
            None,
            input_steps,
            output_steps,
            resolution,
            index_entries=index_entries,
            prefix_frames=prefix_frames,
            cache_trajectories=cache_trajectories,
            max_cache_trajectories=max_cache_trajectories,
        )
    return OfficialArrowWindowDataset(
        arrow_dir,
        index_root / f"{split}_index_{suffix}.json" if index_root else None,
        input_steps,
        output_steps,
        resolution,
        test_mode=mode,
        metadata_root=index_root.parent if index_root else None,
        prefix_frames=prefix_frames,
        cache_trajectories=cache_trajectories,
        max_cache_trajectories=max_cache_trajectories,
    )


def _official_dataset(
    arrow_dir,
    index_root,
    suffix,
    split,
    input_steps,
    output_steps,
    resolution,
    test_mode="all",
    prefix_frames=None,
    *,
    tensor_dir=None,
    cache_trajectories=True,
    max_cache_trajectories=8,
    in_memory=False,
    mmap=True,
):
    return _build_dataset(
        arrow_dir=arrow_dir,
        index_root=index_root,
        suffix=suffix,
        split=split,
        input_steps=input_steps,
        output_steps=output_steps,
        resolution=resolution,
        test_mode=test_mode,
        prefix_frames=prefix_frames,
        tensor_dir=tensor_dir,
        cache_trajectories=cache_trajectories,
        max_cache_trajectories=max_cache_trajectories,
        in_memory=in_memory,
    )


def _setup_normalizer(data_cfg: dict, dataset, device: torch.device, logger) -> GaussianNormalizer | None:
    normalizer_type = str(data_cfg.get("normalizer", "none")).lower().strip()
    if normalizer_type == "gaussian":
        logger.info("Computing Gaussian channel normalizer from training data...")
        normalizer = GaussianNormalizer.from_dataset(
            dataset,
            device=device,
            max_samples=int(data_cfg.get("normalizer_samples", 128)),
        )
        logger.info("GaussianNormalizer computed: mean=%s, std=%s", normalizer.mean.tolist(), normalizer.std.tolist())
        return normalizer
    return None


def run_single_config(config_path: Path, args: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data, training, opt, loss, logcfg = (cfg.get(k, {}) for k in ("data", "training", "optimizer", "loss", "logging"))

    # Model resolution
    model_name = args.model or cfg.get("model", {}).get("name") or cfg.get("model_name")
    if not model_name:
        stem = config_path.stem
        m = re.match(r"^(?:\d+_)?(.*?)(?:_config)?$", stem)
        inferred = m.group(1).lower().replace("_", "-") if m else stem
        if inferred in MODELS or inferred.replace("-", "_") in [x.replace("-", "_") for x in MODELS]:
            model_name = inferred
        elif "triad" in inferred:
            model_name = "triad-mno"
        elif "transolver" in inferred:
            if "pod" in inferred:
                model_name = "pod-itransolver" if "itransolver" in inferred else "pod-transolver"
            else:
                model_name = "itransolver"
        elif "itransformer" in inferred:
            model_name = "pod-itransformer"
        else:
            raise ValueError(f"could not determine model name for config {config_path}. Please set model.name in YAML or pass --model.")

    run_name = config_path.stem
    root = Path(args.data_root or cfg.get("data_root") or config_path.parent.parent)
    real_dir, sim_dir, index_root = _resolve_data_layout(root, data)
    out = Path(args.output_dir or cfg.get("output_dir", "artifacts/runs")) / run_name
    seed = int(args.seed if args.seed is not None else training.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device or training.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    resolution = tuple(args.resolution or data.get("resolution", [32, 64]))
    stride = int(args.stride or data.get("stride", 20))
    batch = int(args.batch_size or training.get("batch_size", 2))
    workers = int(args.num_workers if args.num_workers is not None else training.get("num_workers", 0))
    # Each worker owns a separate dataset/cache. For random windows, duplicating
    # full-trajectory caches across workers usually costs more IO/RAM than it saves.
    cache_trajectories = bool(data.get("cache_trajectories", workers <= 1))
    max_cache_trajectories = max(1, int(data.get("max_cache_trajectories", 8)))
    epochs = int(args.epochs or training.get("pretrain_epochs", 2))
    input_steps, output_steps = int(data.get("input_steps", 20)), int(data.get("output_steps", 20))
    raw_prefix = args.prefix_frames
    if raw_prefix is None:
        raw_prefix = data.get("prefix_frames", None)
    if raw_prefix is None:
        raw_prefix = data.get("prefix_ratio", None)
    if raw_prefix is None:
        raw_prefix = data.get("prefix_fraction", None)
    if raw_prefix is not None:
        val = float(raw_prefix)
        prefix_frames = val if val > 0 else None
    else:
        prefix_frames = None

    out.mkdir(parents=True, exist_ok=True)
    logger = make_logger(out / "logs")
    logger.info(
        "run_name=%s model=%s device=%s resolution=%s batch_size=%d num_workers=%d cache_trajectories=%s "
        "prefix_frames=%s real_dir=%s sim_dir=%s",
        run_name,
        model_name,
        device,
        resolution,
        batch,
        workers,
        cache_trajectories,
        prefix_frames or "full",
        real_dir,
        sim_dir,
    )

    use_tensor_cache = bool(data.get("use_tensor_cache", True))
    reload_tensors = bool(data.get("reload_tensors", False))
    in_memory = bool(data.get("in_memory", False))
    mmap = bool(data.get("mmap", not in_memory))

    tensor_root_cfg = data.get("tensor_dir")
    if tensor_root_cfg:
        tensor_root = Path(tensor_root_cfg)
    else:
        parent = real_dir.parent.parent if real_dir.parent.name in ("hf_dataset", "real") else real_dir.parent
        tensor_root = parent / f"tensor_cache_{resolution[0]}x{resolution[1]}"

    real_tensor_dir = tensor_root / "real"
    sim_tensor_dir = tensor_root / "numerical"

    if use_tensor_cache:
        has_real_pt = real_tensor_dir.exists() and bool(list(real_tensor_dir.glob("*.pt")))
        has_sim_pt = sim_tensor_dir.exists() and bool(list(sim_tensor_dir.glob("*.pt")))
        if reload_tensors or not (has_real_pt and has_sim_pt):
            logger.info("Preparing tensor cache at %s (reload_tensors=%s) ...", tensor_root, reload_tensors)
            import os
            num_proc = max(1, (os.cpu_count() or 4) // 2)
            preprocess_domain(real_dir, real_tensor_dir, resolution, prefix_frames=None, overwrite=reload_tensors, num_workers=num_proc, desc="Precompute Real Tensors")
            preprocess_domain(sim_dir, sim_tensor_dir, resolution, prefix_frames=None, overwrite=reload_tensors, num_workers=num_proc, desc="Precompute Sim Tensors")
        logger.info("Using fast tensor cache: real=%s sim=%s (in_memory=%s, mmap=%s)", real_tensor_dir, sim_tensor_dir, in_memory, mmap)

    official = bool(index_root and data.get("use_official_indices", False))
    bases = None
    test_mode = args.test_mode or data.get("test_mode", "all")
    split_mode = str(data.get("split_mode", "trajectory_prefix"))
    if split_mode not in {"setting", "official_index", "trajectory_prefix"}:
        raise ValueError(f"unsupported split_mode={split_mode!r}; choose setting, official_index, or trajectory_prefix")

    if split_mode == "trajectory_prefix":
        sim_ids = _load_index_trajectory_ids(index_root, "numerical") if index_root else {}
        real_ids = _load_index_trajectory_ids(index_root, "real") if index_root else {}
        sim_ds_entries = build_trajectory_prefix_entries(
            sim_dir, input_steps, output_steps, stride,
            prefix_frames=prefix_frames,
            trajectory_splits=sim_ids or None, seed=seed,
            train_fraction=float(data.get("train_fraction", .8)),
            val_fraction=float(data.get("val_fraction", .1)),
        )
        real_ds_entries = build_trajectory_prefix_entries(
            real_dir, input_steps, output_steps, stride,
            prefix_frames=prefix_frames,
            trajectory_splits=real_ids or None, seed=seed,
            train_fraction=float(data.get("train_fraction", .8)),
            val_fraction=float(data.get("val_fraction", .1)),
        )
        sim_ds = {
            s: _build_dataset(
                arrow_dir=sim_dir,
                index_root=None,
                suffix=None,
                split=s,
                input_steps=input_steps,
                output_steps=output_steps,
                resolution=resolution,
                index_entries=sim_ds_entries[s],
                prefix_frames=prefix_frames,
                tensor_dir=sim_tensor_dir if use_tensor_cache else None,
                cache_trajectories=cache_trajectories,
                max_cache_trajectories=max_cache_trajectories,
                in_memory=in_memory,
                mmap=mmap,
            )
            for s in ("train", "val", "test")
        }
        real_ds = {
            s: _build_dataset(
                arrow_dir=real_dir,
                index_root=None,
                suffix=None,
                split=s,
                input_steps=input_steps,
                output_steps=output_steps,
                resolution=resolution,
                index_entries=real_ds_entries[s],
                prefix_frames=prefix_frames,
                tensor_dir=real_tensor_dir if use_tensor_cache else None,
                cache_trajectories=cache_trajectories,
                max_cache_trajectories=max_cache_trajectories,
                in_memory=in_memory,
                mmap=mmap,
            )
            for s in ("train", "val", "test")
        }
        if not sim_ids:
            sim_ids = {s: sorted({str(item["sim_id"]) for item in sim_ds_entries[s]}) for s in ("train", "val", "test")}
        if not real_ids:
            real_ids = {s: sorted({str(item["sim_id"]) for item in real_ds_entries[s]}) for s in ("train", "val", "test")}
        manifest = {
            "split_mode": "trajectory_prefix",
            "split_source": "official_index" if index_root and _load_index_trajectory_ids(index_root, "real") else "seeded_filename_split",
            "prefix_frames": prefix_frames,
            "sim_trajectory_ids": sim_ids,
            "real_trajectory_ids": real_ids,
            "sim_windows": {k: len(v) for k, v in sim_ds.items()},
            "real_windows": {k: len(v) for k, v in real_ds.items()},
            "resolution": list(resolution), "stride": stride,
        }
        use_joint_basis = bool(data.get("use_joint_basis", False))
        manifest["use_joint_basis"] = use_joint_basis
        normalizer_type = str(data.get("normalizer", "none")).lower().strip()
        normalizer = _setup_normalizer(data, sim_ds["train"], device, logger)
        manifest["normalizer"] = normalizer_type
        if model_name.startswith("pod-") or "triad" in model_name:
            if use_joint_basis:
                logger.info("Fitting balanced joint POD bases from sim_train + real_train...")
                bases = fit_joint_pod_bases_from_datasets(
                    sim_ds["train"],
                    real_ds["train"],
                    int(data.get("pod_rank", 32)),
                    max_samples_per_domain=int(data.get("pod_fit_samples", 64)),
                    normalizer=normalizer,
                )
            else:
                bases = fit_pod_bases_from_dataset(
                    sim_ds["train"],
                    int(data.get("pod_rank", 32)),
                    max_samples=int(data.get("pod_fit_samples", 64)),
                    normalizer=normalizer,
                )
    elif official and split_mode == "official_index":
        sim_ds = {
            s: _official_dataset(
                sim_dir,
                index_root,
                "numerical",
                s,
                input_steps,
                output_steps,
                resolution,
                test_mode,
                prefix_frames=prefix_frames,
                tensor_dir=sim_tensor_dir if use_tensor_cache else None,
                cache_trajectories=cache_trajectories,
                max_cache_trajectories=max_cache_trajectories,
                in_memory=in_memory,
                mmap=mmap,
            )
            for s in ("train", "val", "test")
        }
        real_ds = {
            s: _official_dataset(
                real_dir,
                index_root,
                "real",
                s,
                input_steps,
                output_steps,
                resolution,
                test_mode,
                prefix_frames=prefix_frames,
                tensor_dir=real_tensor_dir if use_tensor_cache else None,
                cache_trajectories=cache_trajectories,
                max_cache_trajectories=max_cache_trajectories,
                in_memory=in_memory,
                mmap=mmap,
            )
            for s in ("train", "val", "test")
        }
        manifest = {"split_mode": "official_index", "prefix_frames": prefix_frames, "test_mode": test_mode, "index_root": str(index_root), "real_dir": str(real_dir), "sim_dir": str(sim_dir), "sim_windows": {k: len(v) for k, v in sim_ds.items()}, "real_windows": {k: len(v) for k, v in real_ds.items()}, "resolution": list(resolution), "stride": stride}
        use_joint_basis = bool(data.get("use_joint_basis", False))
        manifest["use_joint_basis"] = use_joint_basis
        normalizer_type = str(data.get("normalizer", "none")).lower().strip()
        normalizer = _setup_normalizer(data, sim_ds["train"], device, logger)
        manifest["normalizer"] = normalizer_type
        if model_name.startswith("pod-") or "triad" in model_name:
            if use_joint_basis:
                logger.info("Fitting balanced joint POD bases from sim_train + real_train...")
                bases = fit_joint_pod_bases_from_datasets(
                    sim_ds["train"],
                    real_ds["train"],
                    int(data.get("pod_rank", 32)),
                    max_samples_per_domain=int(data.get("pod_fit_samples", 64)),
                    normalizer=normalizer,
                )
            else:
                bases = fit_pod_bases_from_dataset(
                    sim_ds["train"],
                    int(data.get("pod_rank", 32)),
                    max_samples=int(data.get("pod_fit_samples", 64)),
                    normalizer=normalizer,
                )
    elif split_mode == "official_index":
        raise ValueError("split_mode='official_index' requires the complete official index files")
    else:
        sim = discover_trajectories(sim_dir, "sim")
        real = discover_trajectories(real_dir, "real")
        sim_train, sim_val = split_settings(sim, int(data.get("val_settings", 1)), seed)
        real_train, real_val = split_settings(real, int(data.get("val_settings", 1)), seed)
        from ..data.arrow_dataset import ArrowWindowDataset
        sim_ds = {"train": ArrowWindowDataset(sim_train, input_steps, output_steps, stride, resolution), "val": ArrowWindowDataset(sim_val, input_steps, output_steps, stride, resolution)}
        real_ds = {"train": ArrowWindowDataset(real_train, input_steps, output_steps, stride, resolution), "val": ArrowWindowDataset(real_val, input_steps, output_steps, stride, resolution)}
        use_joint_basis = bool(data.get("use_joint_basis", False))
        normalizer_type = str(data.get("normalizer", "none")).lower().strip()
        normalizer = _setup_normalizer(data, sim_ds["train"], device, logger)
        manifest = {"split_mode": "setting_fallback", "prefix_frames": prefix_frames, "use_joint_basis": use_joint_basis, "normalizer": normalizer_type, "sim_train": [x.sim_id for x in sim_train], "sim_val": [x.sim_id for x in sim_val], "real_train": [x.sim_id for x in real_train], "real_val": [x.sim_id for x in real_val], "resolution": list(resolution), "stride": stride}
        if model_name.startswith("pod-") or "triad" in model_name:
            if use_joint_basis:
                logger.info("Fitting balanced joint POD bases from sim_train + real_train...")
                bases = fit_joint_pod_bases_from_datasets(
                    sim_ds["train"],
                    real_ds["train"],
                    int(data.get("pod_rank", 32)),
                    max_samples_per_domain=int(data.get("pod_fit_samples", 64)),
                    normalizer=normalizer,
                )
            else:
                bases = fit_pod_bases(sim_train, int(data.get("pod_rank", 32)), resolution, normalizer=normalizer)
    manifest["seed"] = seed
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    norm_payload = {"resolution": list(resolution), "channels": ["u", "v"], "normalizer_type": normalizer_type}
    if normalizer is not None and hasattr(normalizer, "mean") and normalizer.mean is not None:
        norm_payload["mean"] = normalizer.mean.detach().cpu()
        norm_payload["std"] = normalizer.std.detach().cpu()
    torch.save(norm_payload, out / "normalization.pt")
    if bases is not None:
        basis_dir = out / "pod_basis"
        basis_dir.mkdir(exist_ok=True)
        bases[0].save(basis_dir / "pod_u.npz")
        bases[1].save(basis_dir / "pod_v.npz")

    refit_real_basis = bool(data.get("refit_real_basis", False))
    real_bases = bases
    if refit_real_basis and (model_name.startswith("pod-") or "triad" in model_name):
        real_bases = fit_pod_bases_from_dataset(
            real_ds["train"], int(data.get("pod_rank", 32)), max_samples=int(data.get("pod_fit_samples", 64)), normalizer=normalizer
        )
        real_basis_dir = out / "pod_basis_real"
        real_basis_dir.mkdir(exist_ok=True)
        real_bases[0].save(real_basis_dir / "pod_u.npz")
        real_bases[1].save(real_basis_dir / "pod_v.npz")

    def run(stage, train_ds, val_ds, stage_epochs, stage_bases, initial=None):
        stage_dir = out / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        (stage_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        torch.save(norm_payload, stage_dir / "normalization.pt")
        if stage_bases is not None:
            stage_bases[0].save(stage_dir / "pod_u.npz")
            stage_bases[1].save(stage_dir / "pod_v.npz")
        stage_logger = make_logger(stage_dir / "logs")
        params = {
            "model": model_name,
            "batch_size": batch,
            "num_workers": workers,
            "persistent_workers": bool(training.get("persistent_workers", workers > 0)),
            "prefetch_factor": max(1, int(training.get("prefetch_factor", 1))),
            "non_blocking": bool(training.get("non_blocking", True)),
            "progress_sync": bool(training.get("progress_sync", False)),
            "epochs": stage_epochs,
            "patience": int(training.get("patience", 10)),
            "lr": float(opt.get("lr", 2e-4)),
            "weight_decay": float(opt.get("weight_decay", 1e-4)),
            "grad_clip": float(training.get("grad_clip", 1.0)),
            "field_weight": float(loss.get("field_weight", 0.0)),
            "tke_weight": float(loss.get("tke_weight", 0.05)),
            "use_tke": loss.get("use_tke", None),
            "vorticity_weight": float(loss.get("vorticity_weight", 0.1)),
            "use_vorticity": loss.get("use_vorticity", None),
            "use_amp": bool(training.get("use_amp", True)),
            "save_epoch_checkpoints": bool(logcfg.get("save_epoch_checkpoints", True)),
            "logger": stage_logger,
        }
        if args.resume and stage == ("finetune_real" if args.resume.parent.name == "finetune_real" else "pretrain_sim"):
            params["resume"] = args.resume
        model_options = dict(cfg.get("model", {}))
        model_options.pop("width", None)
        model_options.pop("name", None)
        model_options.setdefault("resolution", tuple(resolution))
        model = build_model(
            model_name,
            stage_bases,
            width=int(cfg.get("model", {}).get("width", 32)),
            input_steps=input_steps,
            output_steps=output_steps,
            **model_options,
        )
        if stage == "finetune_real" and bool(training.get("freeze_macro", False)):
            if hasattr(model, "freeze_macro"):
                model.freeze_macro(True)
                stage_logger.info("Strategy A active: Macro physical backbone frozen for finetune_real")
        return train_stage(model, train_ds, val_ds, stage_dir, params, stage, device, initial, normalizer=normalizer)

    sim_epochs = int(args.epochs) if args.epochs is not None else int(training.get("pretrain_epochs", epochs))
    real_epochs = int(args.epochs) if args.epochs is not None else int(training.get("finetune_epochs", epochs))
    try:
        sim_state, best_sim, sim_best_epoch = run("pretrain_sim", sim_ds["train"], sim_ds["val"], sim_epochs, bases)
        # Release simulation dataset memory and page cache before finetuning real data
        del sim_ds
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        real_state, best_real, real_best_epoch = run("finetune_real", real_ds["train"], real_ds["val"], real_epochs, real_bases, sim_state)
    except Exception:
        logger.exception("training failed for model=%s run=%s", model_name, run_name)
        raise

    result = {
        "run_name": run_name,
        "model": model_name,
        "device": str(device),
        "resolution": list(resolution),
        "prefix_frames": prefix_frames,
        "split_mode": manifest["split_mode"],
        "sim_best_val_loss": best_sim,
        "sim_best_epoch": sim_best_epoch,
        "real_best_val_loss": best_real,
        "real_best_epoch": real_best_epoch,
        "output_dir": str(out),
    }

    eval_split = "test" if "test" in real_ds else "val"
    if eval_split in real_ds:
        model_options = dict(cfg.get("model", {}))
        model_options.pop("width", None)
        model_options.pop("name", None)
        model_options.setdefault("resolution", tuple(resolution))
        test_model = build_model(
            model_name,
            real_bases,
            width=int(cfg.get("model", {}).get("width", 32)),
            input_steps=input_steps,
            output_steps=output_steps,
            **model_options,
        )
        test_model.load_state_dict(real_state, strict=False)

        primary_metrics = evaluate_model(test_model, real_ds[eval_split], device, batch, workers, normalizer=normalizer)
        result[f"real_{eval_split}_mse"] = primary_metrics["mse"]
        result[f"real_{eval_split}_metrics"] = primary_metrics
        result[f"real_{eval_split}_windows"] = len(real_ds[eval_split])

        if official and index_root:
            subsets = ["all", "in_dist", "out_dist", "seen"]
            subsets_metrics = {}
            for sub in subsets:
                try:
                    sub_ds = _official_dataset(
                        real_dir,
                        index_root,
                        "real",
                        "test",
                        input_steps,
                        output_steps,
                        resolution,
                        test_mode=sub,
                        prefix_frames=prefix_frames,
                        tensor_dir=real_tensor_dir if use_tensor_cache else None,
                        in_memory=in_memory,
                        mmap=mmap,
                    )
                    m = evaluate_model(test_model, sub_ds, device, batch, workers, normalizer=normalizer)
                    m["windows"] = len(sub_ds)
                    subsets_metrics[sub] = m
                except Exception as e:
                    logger.warning("could not evaluate subset %s: %s", sub, e)
            result["official_subsets"] = subsets_metrics

            header = f"\n{'='*80}\n{'Subset':<12} | {'MSE':<12} | {'RMSE':<12} | {'MAE':<12} | {'Rel-L2':<12} | {'Windows':<8}\n{'-'*80}"
            rows = []
            for sub, m in subsets_metrics.items():
                rows.append(
                    f"{sub:<12} | {m['mse']:<12.6g} | {m['rmse']:<12.6g} | {m['mae']:<12.6g} | {m['rel_l2']:<12.6g} | {m['windows']:<8}"
                )
            table_str = header + "\n" + "\n".join(rows) + f"\n{'='*80}"
            logger.info(table_str)

    (out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    test_summary = f" test_mse={result['real_test_mse']:.6g}" if "real_test_mse" in result else ""
    if "real_test_metrics" in result:
        test_summary += f" test_rel_l2={result['real_test_metrics']['rel_l2']:.6g}"
    logger.info("completed %s: sim=%.6g real=%.6g%s", run_name, best_sim, best_real, test_summary)

    # -------------------------------------------------------------
    # Archive best checkpoint, logs, configs, and bases
    # -------------------------------------------------------------
    best_checkpoints_root = Path(getattr(args, "best_checkpoints_dir", None) or cfg.get("best_checkpoints_dir", "best_checkpoints")) / run_name
    best_checkpoints_root.mkdir(parents=True, exist_ok=True)

    if (out / "finetune_real" / "best.pt").exists():
        shutil.copy2(out / "finetune_real" / "best.pt", best_checkpoints_root / "best.pt")
    if (out / "pretrain_sim" / "best.pt").exists():
        shutil.copy2(out / "pretrain_sim" / "best.pt", best_checkpoints_root / "best_sim.pt")

    logs_src = out / "logs"
    if logs_src.exists():
        logs_dst = best_checkpoints_root / "logs"
        logs_dst.mkdir(exist_ok=True)
        for log_f in logs_src.glob("*.log"):
            shutil.copy2(log_f, logs_dst / log_f.name)
        if (logs_src / "train.log").exists():
            shutil.copy2(logs_src / "train.log", best_checkpoints_root / "train.log")

    for stage_name in ("pretrain_sim", "finetune_real"):
        stage_dir = out / stage_name
        if stage_dir.exists():
            stage_dst = best_checkpoints_root / stage_name
            stage_dst.mkdir(exist_ok=True)
            for fname in ("history.json", "metrics.jsonl"):
                if (stage_dir / fname).exists():
                    shutil.copy2(stage_dir / fname, stage_dst / fname)
            if (stage_dir / "logs" / "train.log").exists():
                (stage_dst / "logs").mkdir(exist_ok=True)
                shutil.copy2(stage_dir / "logs" / "train.log", stage_dst / "logs" / "train.log")

    if (out / "config.yaml").exists():
        shutil.copy2(out / "config.yaml", best_checkpoints_root / "config.yaml")
    if (out / "results.json").exists():
        shutil.copy2(out / "results.json", best_checkpoints_root / "results.json")
    if (out / "split_manifest.json").exists():
        shutil.copy2(out / "split_manifest.json", best_checkpoints_root / "split_manifest.json")
    if (out / "normalization.pt").exists():
        shutil.copy2(out / "normalization.pt", best_checkpoints_root / "normalization.pt")

    if (out / "pod_basis" / "pod_u.npz").exists():
        shutil.copy2(out / "pod_basis" / "pod_u.npz", best_checkpoints_root / "pod_u.npz")
        shutil.copy2(out / "pod_basis" / "pod_v.npz", best_checkpoints_root / "pod_v.npz")
    if (out / "pod_basis_real" / "pod_u.npz").exists():
        shutil.copy2(out / "pod_basis_real" / "pod_u.npz", best_checkpoints_root / "pod_u_real.npz")
        shutil.copy2(out / "pod_basis_real" / "pod_v.npz", best_checkpoints_root / "pod_v_real.npz")

    patience = int(training.get("patience", 10))
    info_lines = [
        f"Run Name: {run_name}",
        f"Model: {model_name}",
        f"Prefix Frames: {prefix_frames}",
        f"Joint POD Basis: {use_joint_basis}",
        f"Finetune Real Best Epoch: {real_best_epoch}",
        f"Finetune Real Best Val Loss: {best_real:.6g}",
        f"Early Stopping Patience: {patience}",
        f"Pretrain Sim Best Epoch: {sim_best_epoch}",
        f"Pretrain Sim Best Val Loss: {best_sim:.6g}",
    ]
    if "real_test_mse" in result:
        info_lines.append(f"Test MSE: {result['real_test_mse']:.6g}")
    if "real_test_metrics" in result:
        m = result["real_test_metrics"]
        info_lines.append(f"Test Rel-L2: {m.get('rel_l2', 'N/A')}")
        info_lines.append(f"Test RMSE: {m.get('rmse', 'N/A')}")
        info_lines.append(f"Test MAE: {m.get('mae', 'N/A')}")
        if "vorticity_rel_l2" in m:
            info_lines.append(f"Test Vorticity Rel-L2: {m.get('vorticity_rel_l2', 'N/A')}")
    (best_checkpoints_root / "info.txt").write_text("\n".join(info_lines) + "\n", encoding="utf-8")
    logger.info("archived best checkpoint and logs to %s", best_checkpoints_root)

    PrecomputedTrajectoryDataset.clear_cache()
    import gc
    gc.collect()

    return result


def parse_gpu_ids(gpu_arg: str | None) -> list[str]:
    """Parse a GPU argument string into a list of GPU ID strings.

    Supports:
        - None or empty string -> []
        - 'auto' or 'all' -> all available CUDA devices (e.g. ['0', '1', '2', '3'])
        - Comma-separated list -> '0,1,2,3' -> ['0', '1', '2', '3']
        - Range -> '0-3' -> ['0', '1', '2', '3']
        - Mixed -> '0, 1-3' -> ['0', '1', '2', '3']
    """
    if not gpu_arg:
        return []
    gpu_arg = str(gpu_arg).strip()
    if gpu_arg.lower() in ("auto", "all"):
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if count == 0:
            return ["0"]
        return [str(i) for i in range(count)]

    gpu_ids = []
    for part in gpu_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            gpu_ids.extend(str(i) for i in range(int(start.strip()), int(end.strip()) + 1))
        else:
            gpu_ids.append(str(int(part)))

    seen = set()
    deduped = []
    for gid in gpu_ids:
        if gid not in seen:
            seen.add(gid)
            deduped.append(gid)
    return deduped


def ensure_tensor_cache_ready(config_files: list[Path], args: argparse.Namespace) -> None:
    """Precompute tensor cache once in the parent process if required, avoiding worker write races."""
    for cfg_file in config_files:
        try:
            cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        data = cfg.get("data", {})
        if not bool(data.get("use_tensor_cache", True)):
            continue
        reload_tensors = bool(data.get("reload_tensors", False))
        data_root = Path(getattr(args, "data_root", None) or cfg.get("data_root") or cfg_file.parent.parent)
        resolution = tuple(getattr(args, "resolution", None) or data.get("resolution", (64, 128)))
        try:
            real_dir, sim_dir, _ = _resolve_data_layout(data_root, data)
        except Exception:
            continue

        tensor_root_cfg = data.get("tensor_dir")
        if tensor_root_cfg:
            tensor_root = Path(tensor_root_cfg)
        else:
            parent = real_dir.parent.parent if real_dir.parent.name in ("hf_dataset", "real") else real_dir.parent
            tensor_root = parent / f"tensor_cache_{resolution[0]}x{resolution[1]}"

        real_tensor_dir = tensor_root / "real"
        sim_tensor_dir = tensor_root / "numerical"
        has_real_pt = real_tensor_dir.exists() and bool(list(real_tensor_dir.glob("*.pt")))
        has_sim_pt = sim_tensor_dir.exists() and bool(list(sim_tensor_dir.glob("*.pt")))

        if reload_tensors or not (has_real_pt and has_sim_pt):
            print(f"[Multi-GPU Dispatcher] Precomputing tensor cache at {tensor_root} before parallel dispatch...")
            num_proc = max(1, (os.cpu_count() or 4) // 2)
            preprocess_domain(real_dir, real_tensor_dir, resolution, prefix_frames=None, overwrite=reload_tensors, num_workers=num_proc, desc="Precompute Real Tensors")
            preprocess_domain(sim_dir, sim_tensor_dir, resolution, prefix_frames=None, overwrite=reload_tensors, num_workers=num_proc, desc="Precompute Sim Tensors")
            print(f"[Multi-GPU Dispatcher] Tensor cache ready at {tensor_root}.")
            return


def run_configs_multi_gpu(
    config_files: list[Path],
    gpu_ids: list[str],
    args: argparse.Namespace,
) -> dict[str, dict]:
    """Execute multiple configuration files concurrently across specified GPUs using isolated worker subprocesses."""
    ensure_tensor_cache_ready(config_files, args)

    print(f"\n{'='*80}")
    print(f"Multi-GPU Training Dispatcher: {len(config_files)} configurations, {len(gpu_ids)} GPUs: {gpu_ids}")
    print(f"{'='*80}\n")

    pending_configs = list(config_files)
    idle_gpus = list(gpu_ids)
    active_jobs: dict[str, dict] = {}
    completed_results: dict[str, dict] = {}
    last_heartbeat = time.time()

    def _build_cmd(cfg_path: Path) -> list[str]:
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "pod_sim2real.training.train",
            "--config",
            str(cfg_path),
        ]
        if args.model:
            cmd.extend(["--model", str(args.model)])
        if args.data_root:
            cmd.extend(["--data-root", str(args.data_root)])
        if args.output_dir:
            cmd.extend(["--output-dir", str(args.output_dir)])
        if args.best_checkpoints_dir:
            cmd.extend(["--best-checkpoints-dir", str(args.best_checkpoints_dir)])
        if args.resolution:
            cmd.extend(["--resolution", str(args.resolution[0]), str(args.resolution[1])])
        if args.stride is not None:
            cmd.extend(["--stride", str(args.stride)])
        if args.epochs is not None:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.batch_size is not None:
            cmd.extend(["--batch-size", str(args.batch_size)])
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])
        if args.num_workers is not None:
            cmd.extend(["--num-workers", str(args.num_workers)])
        if args.resume:
            cmd.extend(["--resume", str(args.resume)])
        if args.test_mode:
            cmd.extend(["--test-mode", str(args.test_mode)])
        if args.prefix_frames is not None:
            cmd.extend(["--prefix-frames", str(args.prefix_frames)])
        if args.device:
            cmd.extend(["--device", str(args.device)])
        return cmd

    try:
        while pending_configs or active_jobs:
            # Launch configs on available GPUs
            while idle_gpus and pending_configs:
                gpu_id = idle_gpus.pop(0)
                cfg_path = pending_configs.pop(0)

                run_name = cfg_path.stem
                try:
                    cfg_dict = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                    base_out = args.output_dir or cfg_dict.get("output_dir", "artifacts/runs")
                except Exception:
                    base_out = args.output_dir or "artifacts/runs"
                out_dir = Path(base_out) / run_name

                out_dir.mkdir(parents=True, exist_ok=True)
                console_log_path = out_dir / "console.log"
                log_fp = open(console_log_path, "w", encoding="utf-8")

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                env["PYTHONUNBUFFERED"] = "1"
                src_path = str(Path(__file__).resolve().parent.parent.parent)
                if "PYTHONPATH" in env and env["PYTHONPATH"]:
                    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
                else:
                    env["PYTHONPATH"] = src_path

                cmd = _build_cmd(cfg_path)
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    env=env,
                )

                active_jobs[gpu_id] = {
                    "proc": proc,
                    "cfg": cfg_path,
                    "run_name": run_name,
                    "out_dir": out_dir,
                    "log_fp": log_fp,
                    "log_path": console_log_path,
                    "start_time": time.time(),
                }
                now_str = time.strftime("%H:%M:%S")
                print(f"[{now_str}] [GPU {gpu_id}] >>> Started {cfg_path.stem} | Log: {console_log_path}")

            # Check status of running processes
            for gpu_id in list(active_jobs.keys()):
                job = active_jobs[gpu_id]
                ret = job["proc"].poll()
                if ret is not None:
                    job["log_fp"].close()
                    elapsed = time.time() - job["start_time"]
                    m, s = divmod(int(elapsed), 60)
                    elapsed_str = f"{m}m {s:02d}s"
                    now_str = time.strftime("%H:%M:%S")

                    if ret == 0:
                        res_file = job["out_dir"] / "results.json"
                        res_data = {}
                        if res_file.exists():
                            try:
                                res_data = json.loads(res_file.read_text(encoding="utf-8"))
                            except Exception:
                                pass
                        test_mse = res_data.get("real_test_mse")
                        test_rel = res_data.get("real_test_metrics", {}).get("rel_l2")
                        mse_str = f"Test MSE: {test_mse:.6g}" if isinstance(test_mse, (float, int)) else ""
                        rel_str = f"Rel-L2: {test_rel:.6g}" if isinstance(test_rel, (float, int)) else ""
                        metrics_summary = f"{mse_str} {rel_str}".strip()
                        print(f"[{now_str}] [GPU {gpu_id}] [COMPLETED] {job['cfg'].stem} ({elapsed_str}) {metrics_summary}")
                        completed_results[job["cfg"].stem] = {
                            "status": "SUCCESS",
                            "duration_seconds": elapsed,
                            "duration_str": elapsed_str,
                            "results": res_data,
                            "log_path": str(job["log_path"]),
                        }
                    else:
                        print(f"[{now_str}] [GPU {gpu_id}] [FAILED] {job['cfg'].stem} (exit code: {ret}, elapsed: {elapsed_str})")
                        try:
                            lines = job["log_path"].read_text(encoding="utf-8", errors="replace").splitlines()
                            tail = "\n".join(lines[-15:])
                            print(f"--- Tail of {job['log_path']} ---\n{tail}\n---------------------------------------------")
                        except Exception as e:
                            print(f"Could not read log file: {e}")
                        completed_results[job["cfg"].stem] = {
                            "status": f"FAILED (exit {ret})",
                            "duration_seconds": elapsed,
                            "duration_str": elapsed_str,
                            "log_path": str(job["log_path"]),
                        }

                    del active_jobs[gpu_id]
                    idle_gpus.append(gpu_id)

            # Heartbeat print every 30s if jobs are running
            if active_jobs and (time.time() - last_heartbeat >= 30.0):
                last_heartbeat = time.time()
                now_str = time.strftime("%H:%M:%S")
                running_desc = ", ".join(
                    f"GPU {gid}: {j['cfg'].stem} ({int(time.time() - j['start_time'])}s)"
                    for gid, j in active_jobs.items()
                )
                done_count = len(completed_results)
                total_count = len(config_files)
                print(f"[{now_str}] Status: {len(active_jobs)} active, {len(pending_configs)} pending, {done_count}/{total_count} done | {running_desc}")

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[Ctrl+C received] Terminating all active multi-GPU training processes...")
        for gid, job in active_jobs.items():
            try:
                job["proc"].terminate()
            except Exception:
                pass
        time.sleep(1.5)
        for gid, job in active_jobs.items():
            try:
                if job["proc"].poll() is None:
                    job["proc"].kill()
                job["log_fp"].close()
            except Exception:
                pass
        print("[Multi-GPU Dispatcher] All processes terminated cleanly.")
        raise

    # -------------------------------------------------------------
    # Print Final Summary Table
    # -------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"Multi-GPU Training Summary ({len(gpu_ids)} GPUs)")
    print(f"{'='*80}")
    print(f"{'Config Name':<32} | {'Status':<10} | {'Duration':<10} | {'Test MSE':<12} | {'Test Rel-L2':<12}")
    print(f"{'-'*80}")
    for cfg_file in config_files:
        info = completed_results.get(cfg_file.stem, {})
        status = info.get("status", "UNKNOWN")
        dur = info.get("duration_str", "N/A")
        res = info.get("results", {})
        tmse = res.get("real_test_mse")
        trel = res.get("real_test_metrics", {}).get("rel_l2")
        tmse_str = f"{tmse:.6g}" if isinstance(tmse, (float, int)) else "N/A"
        trel_str = f"{trel:.6g}" if isinstance(trel, (float, int)) else "N/A"
        print(f"{cfg_file.stem:<32} | {status:<10} | {dur:<10} | {tmse_str:<12} | {trel_str:<12}")
    print(f"{'='*80}\n")

    # Save summary to artifacts/runs/multi_gpu_summary.json
    summary_path = Path("artifacts/runs/multi_gpu_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(completed_results, indent=2), encoding="utf-8")
    print(f"Detailed run summary saved to {summary_path}")

    return completed_results


def main() -> None:
    parser = argparse.ArgumentParser(description="POD Sim2Real Training Pipeline")
    parser.add_argument("--config", type=Path, help="Path to single YAML configuration file")
    parser.add_argument("--config-dir", type=Path, help="Directory containing YAML configuration files to run")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (e.g. '0,1,2,3'), range ('0-3'), or 'auto'/'all' to run configs in parallel",
    )
    parser.add_argument("--model", choices=MODELS, help="Model name override")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--best-checkpoints-dir", type=Path, default=Path("best_checkpoints"))
    parser.add_argument("--resolution", nargs=2, type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--test-mode", choices=("all", "seen", "in_dist", "out_dist", "unseen"))
    parser.add_argument("--prefix-frames", type=float, help="Fixed number of frames (> 1) or fraction (0 < float <= 1.0) to use from start of each trajectory")
    args = parser.parse_args()

    if args.config_dir:
        config_dir = Path(args.config_dir)
        if not config_dir.exists():
            raise FileNotFoundError(f"config-dir not found: {config_dir}")
        config_files = sorted(config_dir.glob("*.yaml"))
        if not config_files:
            raise FileNotFoundError(f"no YAML configuration files found in {config_dir}")
        print(f"Discovered {len(config_files)} configurations in {config_dir}: {[f.name for f in config_files]}")

        gpu_ids = parse_gpu_ids(args.gpus)
        if len(gpu_ids) > 1:
            run_configs_multi_gpu(config_files, gpu_ids, args)
            return

        if len(gpu_ids) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]

        all_results = {}
        for cfg_file in config_files:
            print(f"\n{'='*80}\nStarting configuration: {cfg_file.name}\n{'='*80}")
            try:
                res = run_single_config(cfg_file, args)
                all_results[cfg_file.stem] = res
            except Exception as e:
                print(f"Error running {cfg_file.name}: {e}")
                raise
        print(f"\nAll {len(config_files)} configurations in {config_dir} completed successfully!")
    elif args.config:
        gpu_ids = parse_gpu_ids(args.gpus)
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        run_single_config(args.config, args)
    else:
        parser.error("must provide either --config or --config-dir")


if __name__ == "__main__":
    main()

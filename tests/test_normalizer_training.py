from pathlib import Path
import pytest
import torch
import yaml

from pod_sim2real.training.train import run_single_config


def test_training_pipeline_with_gaussian_normalizer(tmp_path):
    runs_dir = tmp_path / "runs"
    best_dir = tmp_path / "best"
    cfg_file = tmp_path / "test_normalizer_cfg.yaml"

    cfg = {
        "data": {
            "use_official_indices": False,
            "split_mode": "setting",
            "real_dir": "data/data_real",
            "sim_dir": "data/data_sim",
            "val_settings": 1,
            "input_steps": 20,
            "output_steps": 20,
            "resolution": [32, 64],
            "stride": 500,
            "pod_rank": 8,
            "prefix_frames": 2000,
            "normalizer": "gaussian",
            "normalizer_samples": 8,
            "pod_fit_samples": 8,
        },
        "training": {
            "pretrain_epochs": 1,
            "finetune_epochs": 1,
            "batch_size": 2,
            "patience": 1,
            "num_workers": 0,
            "seed": 42,
            "device": "cpu",
            "use_amp": False,
        },
        "optimizer": {"lr": 0.001, "weight_decay": 0.0},
        "loss": {"use_tke": False, "use_vorticity": False},
        "logging": {"use_tqdm": False, "save_epoch_checkpoints": False},
        "model": {
            "name": "triad-afno",
            "width": 8,
            "depth": 1,
            "heads": 2,
            "macro_rank": 4,
            "micro_rank": 4,
            "mode_layout": "grid2d",
        },
        "output_dir": str(runs_dir),
        "best_checkpoints_dir": str(best_dir),
    }

    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    class MockArgs:
        config = cfg_file
        config_dir = None
        model = None
        data_root = Path.cwd()
        output_dir = runs_dir
        best_checkpoints_dir = best_dir
        resolution = None
        stride = None
        device = "cpu"
        epochs = 1
        lr = None
        batch_size = 2
        seed = 42
        num_workers = 0
        test_mode = None
        prefix_frames = None
        tag = "test_norm"
        resume = None

    result = run_single_config(cfg_file, MockArgs())
    assert result is not None
    assert "sim_best_val_loss" in result
    assert "real_best_val_loss" in result

    # Check normalization.pt
    norm_path = Path(result["output_dir"]) / "normalization.pt"
    assert norm_path.exists()
    payload = torch.load(norm_path, map_location="cpu", weights_only=False)
    assert payload["normalizer_type"] == "gaussian"
    assert "mean" in payload
    assert "std" in payload
    assert payload["mean"].shape == torch.Size([2])
    assert payload["std"].shape == torch.Size([2])

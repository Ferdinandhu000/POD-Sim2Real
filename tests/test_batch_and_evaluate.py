import json
from pathlib import Path
import pandas as pd
import pytest
import yaml

from pod_sim2real.training.train import run_single_config, main
from pod_sim2real.training.evaluate import evaluate_all_checkpoints


def test_batch_training_and_evaluate(tmp_path):
    # Setup two mock smoke configs using legacy data
    cfg_dir = tmp_path / "mock_yamls"
    cfg_dir.mkdir()
    runs_dir = tmp_path / "runs"
    best_dir = tmp_path / "best_checkpoints"

    base_dict = {
        "data": {
            "use_official_indices": False,
            "split_mode": "setting",
            "real_dir": "data/data_real",
            "sim_dir": "data/data_sim",
            "val_settings": 1,
            "input_steps": 20,
            "output_steps": 20,
            "resolution": [32, 64],
            "stride": 400,
            "pod_rank": 8,
            "prefix_frames": 2000,
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
        "output_dir": str(runs_dir),
        "best_checkpoints_dir": str(best_dir),
    }

    # Config 1: unet
    cfg1 = dict(base_dict)
    cfg1["model"] = {"name": "unet", "width": 8}
    (cfg_dir / "00_test_unet_config.yaml").write_text(yaml.safe_dump(cfg1), encoding="utf-8")

    # Config 2: pod-fno
    cfg2 = dict(base_dict)
    cfg2["model"] = {"name": "pod-fno", "width": 8, "modes": 4}
    (cfg_dir / "01_test_podfno_config.yaml").write_text(yaml.safe_dump(cfg2), encoding="utf-8")

    class MockArgs:
        config = None
        config_dir = cfg_dir
        model = None
        data_root = Path.cwd()
        output_dir = runs_dir
        best_checkpoints_dir = best_dir
        resolution = None
        stride = None
        epochs = 1
        batch_size = None
        device = "cpu"
        seed = 42
        num_workers = 0
        resume = None
        test_mode = None
        prefix_frames = None

    # Run single config for both to verify archiving
    res1 = run_single_config(cfg_dir / "00_test_unet_config.yaml", MockArgs())
    assert (best_dir / "00_test_unet_config" / "best.pt").exists()
    assert (best_dir / "00_test_unet_config" / "info.txt").exists()
    assert (best_dir / "00_test_unet_config" / "config.yaml").exists()

    res2 = run_single_config(cfg_dir / "01_test_podfno_config.yaml", MockArgs())
    assert (best_dir / "01_test_podfno_config" / "best.pt").exists()
    assert (best_dir / "01_test_podfno_config" / "pod_u.npz").exists()
    assert (best_dir / "01_test_podfno_config" / "pod_v.npz").exists()

    # Verify info.txt contents
    info_text = (best_dir / "00_test_unet_config" / "info.txt").read_text(encoding="utf-8")
    assert "Run Name: 00_test_unet_config" in info_text
    assert "Best Epoch: 1" in info_text

    # Verify isolation
    assert (runs_dir / "00_test_unet_config").exists()
    assert (runs_dir / "01_test_podfno_config").exists()


def test_triad_mno_macro_only_freeze():
    import numpy as np
    from pod_sim2real.model import PODBasis, TriadMNO

    dummy_modes = np.random.randn(32, 64).astype(np.float32)
    dummy_mean = np.random.randn(64).astype(np.float32)
    basis_u = PODBasis(dummy_mean, dummy_modes)
    basis_v = PODBasis(dummy_mean, dummy_modes)

    # micro_rank = 0
    model = TriadMNO((basis_u, basis_v), macro_rank=32, micro_rank=0, width=16)
    assert model.neural_field is None
    # Calling freeze_macro should be a no-op and not freeze all parameters
    model.freeze_macro(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0, "macro_only model should not have 0 trainable parameters after freeze_macro"


def test_joint_pod_basis_training(tmp_path):
    cfg_dir = tmp_path / "mock_yamls"
    cfg_dir.mkdir()
    runs_dir = tmp_path / "runs"
    best_dir = tmp_path / "best_checkpoints"

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
            "stride": 400,
            "pod_rank": 8,
            "prefix_frames": 2000,
            "use_joint_basis": True,
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
        "model": {"name": "pod-fno", "width": 8, "modes": 4},
        "output_dir": str(runs_dir),
        "best_checkpoints_dir": str(best_dir),
    }

    cfg_path = cfg_dir / "01_test_joint_podfno_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    class MockArgs:
        model = None
        data_root = Path.cwd()
        output_dir = runs_dir
        best_checkpoints_dir = best_dir
        resolution = None
        stride = None
        epochs = 1
        batch_size = None
        device = "cpu"
        seed = 42
        num_workers = 0
        resume = None
        test_mode = None
        prefix_frames = None

    res = run_single_config(cfg_path, MockArgs())
    assert res["model"] == "pod-fno"

    target_best = best_dir / "01_test_joint_podfno_config"
    assert (target_best / "pod_u.npz").exists()
    assert (target_best / "pod_v.npz").exists()
    info_text = (target_best / "info.txt").read_text(encoding="utf-8")
    assert "Joint POD Basis: True" in info_text
    manifest = json.loads((target_best / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("use_joint_basis") is True

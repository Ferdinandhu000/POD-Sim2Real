import numpy as np
import pytest
import torch

from pod_sim2real.model import (
    PODBasis,
    PODTransformer,
    PODTransolver,
    PODiTransolver,
    Transformer3d,
    Transolver3d,
    build_model,
    iTransolver3d,
)


@pytest.fixture
def dummy_input_3d():
    # [Batch, T_in, C, H, W]
    return torch.randn(2, 4, 2, 16, 32)


@pytest.fixture
def dummy_pod_bases():
    # Resolution (16, 32) -> 512 points, rank 8
    points = 16 * 32
    rank = 8
    mean_u = np.zeros(points, dtype=np.float32)
    modes_u = np.random.randn(rank, points).astype(np.float32)
    mean_v = np.zeros(points, dtype=np.float32)
    modes_v = np.random.randn(rank, points).astype(np.float32)
    return (PODBasis(mean_u, modes_u), PODBasis(mean_v, modes_v))


def test_transformer3d_forward_and_backward(dummy_input_3d):
    model = Transformer3d(
        channels=2,
        width=16,
        layers=2,
        heads=2,
        patch_size=(4, 4),
        input_steps=4,
        output_steps=4,
        resolution=(16, 32),
    )
    out = model(dummy_input_3d)
    assert out.shape == dummy_input_3d.shape
    loss = out.sum()
    loss.backward()
    assert model.core.head[1].weight.grad is not None


def test_transolver3d_forward_and_backward(dummy_input_3d):
    model = Transolver3d(
        channels=2,
        width=16,
        layers=2,
        heads=2,
        slice_num=8,
        patch_size=(2, 2),
        input_steps=4,
        output_steps=4,
        resolution=(16, 32),
    )
    out = model(dummy_input_3d)
    assert out.shape == dummy_input_3d.shape
    loss = out.sum()
    loss.backward()
    assert model.core.head[1].weight.grad is not None


def test_itransolver3d_forward_and_backward(dummy_input_3d):
    model = iTransolver3d(
        channels=2,
        width=16,
        layers=2,
        heads=2,
        slice_num=8,
        patch_size=(2, 2),
        use_norm=True,
        input_steps=4,
        output_steps=4,
        resolution=(16, 32),
    )
    out = model(dummy_input_3d)
    assert out.shape == dummy_input_3d.shape
    loss = out.sum()
    loss.backward()
    assert model.core.head[1].weight.grad is not None


def test_pod_transformer_forward_and_backward(dummy_pod_bases, dummy_input_3d):
    model = PODTransformer(
        bases=dummy_pod_bases,
        width=16,
        input_steps=4,
        output_steps=4,
        depth=2,
        heads=2,
    )
    out = model(dummy_input_3d)
    assert out.shape == dummy_input_3d.shape
    loss = out.sum()
    loss.backward()
    assert model.net.lift.weight.grad is not None


def test_pod_transolver_forward_and_backward(dummy_pod_bases, dummy_input_3d):
    model = PODTransolver(
        bases=dummy_pod_bases,
        width=16,
        input_steps=4,
        output_steps=4,
        depth=2,
        heads=2,
        slice_num=4,
    )
    out = model(dummy_input_3d)
    assert out.shape == dummy_input_3d.shape
    loss = out.sum()
    loss.backward()
    assert model.net.lift.weight.grad is not None


def test_pod_itransolver_forward_and_backward(dummy_pod_bases, dummy_input_3d):
    model = PODiTransolver(
        bases=dummy_pod_bases,
        width=16,
        input_steps=4,
        output_steps=4,
        depth=2,
        heads=2,
        slice_num=8,
    )
    out = model(dummy_input_3d)
    assert out.shape == dummy_input_3d.shape
    loss = out.sum()
    loss.backward()
    assert model.embed.weight.grad is not None


def test_build_model_factory_new_models(dummy_pod_bases):
    names = [
        "transformer3d",
        "transolver3d",
        "itransolver3d",
        "pod-transformer",
        "pod-transolver",
        "pod-itransolver",
    ]
    for name in names:
        m = build_model(
            name,
            bases=dummy_pod_bases,
            width=16,
            depth=2,
            heads=2,
            input_steps=4,
            output_steps=4,
            resolution=(16, 32),
            patch_size=(2, 2) if "3d" in name else None,
        )
        assert m is not None


def test_training_pipeline_with_itransolver(tmp_path):
    from pod_sim2real.training.train import run_single_config
    import yaml
    from pathlib import Path

    runs_dir = tmp_path / "runs"
    best_dir = tmp_path / "best"
    cfg_file = tmp_path / "test_itransolver_cfg.yaml"

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
            "normalizer": "gaussian",
            "normalizer_samples": 8,
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
        "model": {
            "name": "itransolver3d",
            "width": 8,
            "depth": 1,
            "heads": 2,
            "slice_num": 4,
            "patch_size": [4, 4],
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
        tag = "test_itransolver"
        resume = None

    result = run_single_config(cfg_file, MockArgs())
    assert result is not None
    assert "sim_best_val_loss" in result
    assert "real_best_val_loss" in result


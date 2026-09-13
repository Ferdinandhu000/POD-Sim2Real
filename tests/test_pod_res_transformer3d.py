import numpy as np
import pytest
import torch

from pod_sim2real.model import (
    PODBasis,
    PODResTransformer3d,
    build_model,
)
from pod_sim2real.training.losses import compute_loss


@pytest.fixture
def mock_bases():
    h, w = 32, 64
    n_points = h * w
    rank = 64
    rng = np.random.RandomState(42)
    mean_u = rng.randn(n_points).astype(np.float32)
    mean_v = rng.randn(n_points).astype(np.float32)
    q_u, _ = np.linalg.qr(rng.randn(n_points, rank))
    q_v, _ = np.linalg.qr(rng.randn(n_points, rank))
    basis_u = PODBasis(mean_u, q_u.T.astype(np.float32))
    basis_v = PODBasis(mean_v, q_v.T.astype(np.float32))
    return (basis_u, basis_v)


def test_pod_res_transformer3d_forward_backward(mock_bases):
    model = PODResTransformer3d(
        mock_bases,
        pod_rank=32,
        pod_width=16,
        pod_depth=2,
        res_layers=2,
        res_width=32,
        res_heads=4,
        res_patch_size=(4, 4),
        input_steps=4,
        output_steps=4,
        resolution=(32, 64),
    )
    x = torch.randn(2, 4, 2, 32, 64)
    target = torch.randn(2, 4, 2, 32, 64)

    out = model(x)
    assert out.shape == (2, 4, 2, 32, 64)

    loss, parts = compute_loss(out, target, kind="pod-res-transformer3d", v_weight=3.0, use_vorticity=True)
    loss.backward()

    assert model.embed_pod.weight.grad is not None
    assert model.residual_transformer.patch_embed.weight.grad is not None


def test_pod_res_transformer3d_build_model(mock_bases):
    model = build_model(
        "pod-res-transformer3d",
        bases=mock_bases,
        width=16,
        input_steps=4,
        output_steps=4,
        pod_rank=32,
        pod_depth=2,
        res_layers=2,
        res_width=32,
        res_heads=4,
        patch_size=(4, 4),
        resolution=(32, 64),
    )
    x = torch.randn(2, 4, 2, 32, 64)
    out = model(x)
    assert out.shape == (2, 4, 2, 32, 64)


def test_pod_res_transformer3d_freezing(mock_bases):
    model = PODResTransformer3d(
        mock_bases,
        pod_rank=32,
        pod_width=16,
        pod_depth=2,
        res_layers=2,
        res_width=32,
        res_heads=4,
        res_patch_size=(4, 4),
        input_steps=4,
        output_steps=4,
        resolution=(32, 64),
    )
    # Freeze macro / POD
    model.freeze_macro(True)
    for p in model.embed_pod.parameters():
        assert not p.requires_grad
    for p in model.residual_transformer.parameters():
        assert p.requires_grad

    # Unfreeze macro, freeze residual
    model.freeze_macro(False)
    model.freeze_residual(True)
    for p in model.embed_pod.parameters():
        assert p.requires_grad
    for p in model.residual_transformer.parameters():
        assert not p.requires_grad


def test_pod_res_transformer3d_dual_aux_loss(mock_bases):
    model = PODResTransformer3d(
        mock_bases,
        pod_rank=32,
        pod_width=16,
        pod_depth=2,
        res_layers=2,
        res_width=32,
        res_heads=4,
        res_patch_size=(4, 4),
        input_steps=4,
        output_steps=4,
        resolution=(32, 64),
        return_aux=True,
    )
    x = torch.randn(2, 4, 2, 32, 64)
    target = torch.randn(2, 4, 2, 32, 64)
    out = model(x)
    assert isinstance(out, tuple)
    assert len(out) == 2
    u_final, u_pod = out
    assert u_final.shape == (2, 4, 2, 32, 64)
    assert u_pod.shape == (2, 4, 2, 32, 64)

    loss, parts = compute_loss(out, target, kind="pod-res-transformer3d", v_weight=3.0, pod_weight=0.5, use_vorticity=True)
    loss.backward()
    assert model.embed_pod.weight.grad is not None


def test_pipeline_integration(tmp_path):
    from pathlib import Path
    import yaml
    from pod_sim2real.training.train import run_single_config

    runs_dir = tmp_path / "runs"
    best_dir = tmp_path / "best_checkpoints"

    cfg = {
        "data": {
            "use_official_indices": False,
            "split_mode": "setting",
            "real_dir": "data/data_real",
            "sim_dir": "data/data_sim",
            "val_settings": 1,
            "input_steps": 4,
            "output_steps": 4,
            "resolution": [32, 64],
            "stride": 400,
            "pod_rank": 8,
            "prefix_frames": 2000,
            "use_joint_basis": True,
            "normalizer": "gaussian",
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
            "freeze_macro": True,
            "freeze_residual_epochs": 1,
        },
        "optimizer": {"lr": 0.001, "weight_decay": 0.0},
        "loss": {"use_tke": False, "use_vorticity": True, "v_weight": 2.0, "pod_weight": 0.5},
        "logging": {"use_tqdm": False, "save_epoch_checkpoints": False},
        "output_dir": str(runs_dir),
        "best_checkpoints_dir": str(best_dir),
        "model": {
            "name": "pod-res-transformer3d",
            "pod_rank": 8,
            "pod_width": 16,
            "pod_depth": 2,
            "res_width": 32,
            "res_layers": 2,
            "res_heads": 4,
            "patch_size": [4, 4],
            "return_aux": True,
        },
    }

    cfg_file = tmp_path / "smoke_pod_res_transformer3d.yaml"
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
        epochs = 1
        batch_size = None
        device = "cpu"
        seed = 42
        num_workers = 0
        resume = None
        test_mode = None
        prefix_frames = None

    res = run_single_config(cfg_file, MockArgs())
    assert (best_dir / "smoke_pod_res_transformer3d" / "best.pt").exists()
    assert (runs_dir / "smoke_pod_res_transformer3d" / "finetune_real" / "best.pt").exists()

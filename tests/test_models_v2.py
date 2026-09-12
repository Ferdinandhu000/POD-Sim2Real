import numpy as np
import pytest
import torch

from pod_sim2real.model import (
    AFNO3d,
    FNO3d,
    PODBasis,
    TriadAFNO,
    Unet3d,
    build_model,
)


@pytest.fixture
def mock_bases():
    # 2 bases (u and v), each with 64 modes of resolution 32x64 (2048 points)
    h, w = 32, 64
    n_points = h * w
    rank = 64
    rng = np.random.RandomState(42)
    mean_u = rng.randn(n_points).astype(np.float32)
    mean_v = rng.randn(n_points).astype(np.float32)
    # Orthogonal modes via QR
    q_u, _ = np.linalg.qr(rng.randn(n_points, rank))
    q_v, _ = np.linalg.qr(rng.randn(n_points, rank))
    basis_u = PODBasis(mean_u, q_u.T.astype(np.float32))
    basis_v = PODBasis(mean_v, q_v.T.astype(np.float32))
    return (basis_u, basis_v)


def test_triad_afno_grid2d(mock_bases):
    model = TriadAFNO(
        mock_bases,
        width=16,
        input_steps=4,
        output_steps=4,
        depth=2,
        macro_rank=16,
        micro_rank=16,
        mode_layout="grid2d",
        heads=2,
    )
    x = torch.randn(2, 4, 2, 32, 64)
    out = model(x)
    assert out.shape == (2, 4, 2, 32, 64)
    loss = out.sum()
    loss.backward()
    assert model.embed_macro.weight.grad is not None
    assert model.triad_cross_attn.in_proj_weight.grad is not None


def test_triad_afno_flat1d(mock_bases):
    model = TriadAFNO(
        mock_bases,
        width=16,
        input_steps=4,
        output_steps=4,
        depth=3,
        macro_rank=16,
        micro_rank=16,
        mode_layout="flat1d",
        heads=2,
    )
    x = torch.randn(2, 4, 2, 32, 64)
    out = model(x)
    assert out.shape == (2, 4, 2, 32, 64)
    loss = out.sum()
    loss.backward()
    assert model.head_macro.weight.grad is not None


def test_triad_afno_depth_ablation(mock_bases):
    # Test depth=4
    model = build_model(
        "triad-afno",
        bases=mock_bases,
        width=16,
        input_steps=4,
        output_steps=4,
        depth=4,
        macro_rank=16,
        micro_rank=16,
        mode_layout="grid2d",
        heads=2,
    )
    assert len(model.macro_net) == 4
    assert len(model.micro_net) == 4
    x = torch.randn(2, 4, 2, 32, 64)
    out = model(x)
    assert out.shape == (2, 4, 2, 32, 64)


def test_fno3d():
    model = build_model(
        "fno3d",
        width=16,
        layers=2,
        modes_t=2,
        modes_h=4,
        modes_w=4,
        input_steps=4,
        output_steps=4,
        resolution=(16, 16),
    )
    x = torch.randn(2, 4, 2, 16, 16)
    out = model(x)
    assert out.shape == (2, 4, 2, 16, 16)
    loss = out.sum()
    loss.backward()
    assert model.core.fc0.weight.grad is not None


def test_unet3d():
    model = build_model(
        "unet3d",
        width=16,
        input_steps=4,
        output_steps=4,
        resolution=(16, 16),
        dim=16,
        dim_mults=(1, 2),
    )
    x = torch.randn(2, 4, 2, 16, 16)
    out = model(x)
    assert out.shape == (2, 4, 2, 16, 16)
    loss = out.sum()
    loss.backward()
    assert model.core.init_conv.weight.grad is not None


def test_afno3d():
    model = build_model(
        "afno3d",
        width=16,
        layers=2,
        blocks=4,
        input_steps=4,
        output_steps=4,
        resolution=(16, 16),
    )
    x = torch.randn(2, 4, 2, 16, 16)
    out = model(x)
    assert out.shape == (2, 4, 2, 16, 16)
    loss = out.sum()
    loss.backward()
    assert model.core.fc0.weight.grad is not None


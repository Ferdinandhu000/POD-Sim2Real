import numpy as np
import pytest
import torch

from pod_sim2real.model import PODBasis, PODResUNet3d, build_model


@pytest.fixture
def mock_bases():
    height, width = 32, 64
    n_points = height * width
    rank = 24
    rng = np.random.RandomState(42)
    bases = []
    for _ in range(2):
        mean = rng.randn(n_points).astype(np.float32)
        q, _ = torch.linalg.qr(torch.from_numpy(rng.randn(n_points, rank).astype(np.float32)))
        bases.append(PODBasis(mean, q.T.numpy().astype(np.float32)))
    return tuple(bases)


def make_model(bases, **kwargs):
    options = {
        "pod_rank": 16,
        "pod_width": 8,
        "pod_depth": 1,
        "res_width": 8,
        "res_dim_mults": (1, 2),
        "res_attn_heads": 1,
        "res_attn_dim_head": 4,
        "resnet_groups": 4,
        "input_steps": 4,
        "output_steps": 4,
        "resolution": (32, 64),
        "return_aux": False,
    }
    options.update(kwargs)
    return PODResUNet3d(bases, **options)


@pytest.mark.parametrize("mode", ["temporal_concat", "future_refine", "channel_concat"])
def test_pod_res_unet3d_forward_backward_modes(mock_bases, mode):
    model = make_model(mock_bases, dual_stream_mode=mode)
    x = torch.randn(2, 4, 2, 32, 64)
    output = model(x)
    assert output.shape == (2, 4, 2, 32, 64)

    loss = output.square().mean()
    loss.backward()
    assert model.embed_pod.weight.grad is not None
    assert model.residual_unet.init_conv.weight.grad is not None
    if mode == "temporal_concat":
        assert model.time_project.weight.grad is not None


def test_pod_res_unet3d_build_model(mock_bases):
    model = build_model(
        "pod-res-unet3d",
        bases=mock_bases,
        width=8,
        input_steps=4,
        output_steps=4,
        pod_rank=16,
        pod_width=8,
        pod_depth=1,
        res_width=8,
        res_dim_mults=(1, 2),
        res_attn_heads=1,
        res_attn_dim_head=4,
        resnet_groups=4,
        resolution=(32, 64),
        dual_stream_mode="temporal_concat",
    )
    output = model(torch.randn(1, 4, 2, 32, 64))
    assert output.shape == (1, 4, 2, 32, 64)


def test_pod_res_unet3d_aux_output(mock_bases):
    model = make_model(mock_bases, return_aux=True)
    output = model(torch.randn(1, 4, 2, 32, 64))
    assert isinstance(output, tuple)
    assert len(output) == 2
    assert output[0].shape == output[1].shape == (1, 4, 2, 32, 64)


def test_pod_res_unet3d_freezing(mock_bases):
    model = make_model(mock_bases, dual_stream_mode="temporal_concat")

    model.freeze_macro(True)
    assert all(not p.requires_grad for p in model.embed_pod.parameters())
    assert all(not p.requires_grad for p in model.afno_layers.parameters())
    assert any(p.requires_grad for p in model.residual_unet.parameters())
    assert all(p.requires_grad for p in model.time_project.parameters())

    model.freeze_macro(False)
    model.freeze_residual(True)
    assert all(p.requires_grad for p in model.embed_pod.parameters())
    assert all(not p.requires_grad for p in model.residual_unet.parameters())
    assert all(not p.requires_grad for p in model.time_project.parameters())


def test_pod_res_unet3d_channel_concat_requires_equal_steps(mock_bases):
    with pytest.raises(ValueError, match="input_steps == output_steps"):
        make_model(mock_bases, dual_stream_mode="channel_concat", input_steps=4, output_steps=2)

import numpy as np
import pytest
import torch

from pod_sim2real.model import (
    PODBasis,
    PODModel,
    PODTransolver,
    PODiTransolver,
    Transolver1dCoeff,
    build_coeff_operator,
    build_model,
    iTransolver1dCoeff,
)


def _bases(height=8, width=12, rank=4):
    points = height * width
    modes = np.eye(points, dtype=np.float32)[:rank]
    return tuple(PODBasis(np.zeros(points, dtype=np.float32), modes) for _ in range(2))


@pytest.mark.parametrize("name", ["unet", "fno", "afno", "itransolver", "pod-fno", "pod-afno", "pod-unet", "pod-itransformer", "pod-itransolver", "pod-transolver", "triad-mno"])
def test_models_preserve_field_contract(name):
    bases = _bases() if (name.startswith("pod-") or "triad" in name) else None
    model = build_model(name, bases, width=8, input_steps=3, output_steps=3, depth=1, heads=2, slice_num=4, macro_rank=2, micro_rank=2)
    x = torch.randn(1, 3, 2, 8, 12)
    output = model(x)
    assert output.shape == x.shape
    output.square().mean().backward()


def test_pod_round_trip_and_buffers_move():
    model = build_model("pod-itransformer", _bases(), width=8, input_steps=3, output_steps=3, depth=1, heads=2)
    coefficients = torch.randn(2, 3, 2 * 4)
    x = model.field(coefficients, 8, 12)
    reconstruction = model.field(model.coeff(x), 8, 12)
    assert torch.allclose(reconstruction, x, atol=1e-6)
    assert model.to("cpu").mean.device.type == "cpu"


def test_models_support_distinct_output_horizon():
    x = torch.randn(1, 3, 2, 8, 12)
    for name in ("unet", "itransolver", "pod-unet", "pod-fno", "pod-afno", "pod-itransformer", "pod-itransolver", "pod-transolver", "triad-mno"):
        bases = _bases() if (name.startswith("pod-") or "triad" in name) else None
        model = build_model(name, bases, width=8, input_steps=3, output_steps=2, depth=1, heads=2, slice_num=4, macro_rank=2, micro_rank=2)
        assert model(x).shape == (1, 2, 2, 8, 12)


def test_triad_mno_freeze_macro():
    bases = _bases()
    model = build_model("triad-mno", bases, width=8, input_steps=3, output_steps=3, depth=1, heads=2, macro_rank=2, micro_rank=2)
    model.freeze_macro(True)
    for p in model.macro_net.parameters():
        assert not p.requires_grad
    for p in model.macro_proj.parameters():
        assert not p.requires_grad
    for p in model.neural_field.parameters():
        assert p.requires_grad
    model.freeze_macro(False)
    for p in model.macro_net.parameters():
        assert p.requires_grad


@pytest.mark.parametrize("macro_op,micro_op", [
    ("fno", "fno"),
    ("itransformer", "fno"),
    ("fno", "itransformer"),
    ("itransolver", "itransolver"),
    ("fno", "itransolver"),
    ("mlp", "mlp"),
])
def test_triad_mno_modular_operators(macro_op, micro_op):
    bases = _bases()
    model = build_model(
        "triad-mno", bases, width=8, input_steps=3, output_steps=3,
        depth=1, heads=2, macro_rank=2, micro_rank=2,
        macro_operator=macro_op, micro_operator=micro_op
    )
    x = torch.randn(1, 3, 2, 8, 12)
    out = model(x)
    assert out.shape == (1, 3, 2, 8, 12)
    out.square().mean().backward()


def test_triad_mno_zero_shot_super_resolution():
    bases = _bases(height=8, width=12, rank=4)
    model = build_model(
        "triad-mno", bases, width=8, input_steps=3, output_steps=3,
        depth=1, heads=2, macro_rank=2, micro_rank=2,
        native_resolution=(8, 12)
    )
    x = torch.randn(1, 3, 2, 8, 12)
    # Query 2x super-resolution
    out_2x = model(x, height=16, width=24)
    assert out_2x.shape == (1, 3, 2, 16, 24)
    out_2x.square().mean().backward()


def test_itransolver_coeff_operator_standalone():
    rank = 4
    op = iTransolver1dCoeff(rank=rank, width=16, depth=2, heads=2, slice_num=4, input_steps=5, output_steps=3)
    x = torch.randn(2, 5, 2 * rank)
    out = op(x)
    assert out.shape == (2, 3, 2 * rank)
    out.sum().backward()


def test_transolver_coeff_operator_standalone():
    rank = 4
    op = Transolver1dCoeff(rank=rank, width=16, depth=2, heads=2, slice_num=4, input_steps=5, output_steps=3)
    x = torch.randn(2, 5, 2 * rank)
    out = op(x)
    assert out.shape == (2, 3, 2 * rank)
    out.sum().backward()


def test_pod_transolver_and_itransolver_distinct():
    assert PODTransolver is not PODiTransolver
    bases = _bases(height=8, width=12, rank=4)
    model_inv = PODiTransolver(bases, width=16, input_steps=4, output_steps=4, depth=2, heads=2, slice_num=4)
    model_fwd = PODTransolver(bases, width=16, input_steps=4, output_steps=4, depth=2, heads=2, slice_num=4)

    assert isinstance(model_inv, PODModel) and isinstance(model_inv, PODiTransolver)
    assert isinstance(model_fwd, PODModel) and isinstance(model_fwd, PODTransolver)
    assert model_inv.kind == "itransolver"
    assert model_fwd.kind == "transolver"
    assert isinstance(model_fwd.net, Transolver1dCoeff)

    x = torch.randn(2, 4, 2, 8, 12)
    out_inv = model_inv(x)
    assert out_inv.shape == (2, 4, 2, 8, 12)
    out_inv.sum().backward()

    out_fwd = model_fwd(x)
    assert out_fwd.shape == (2, 4, 2, 8, 12)
    out_fwd.sum().backward()






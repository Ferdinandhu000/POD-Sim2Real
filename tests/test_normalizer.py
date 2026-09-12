import math
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from pod_sim2real.data.normalizer import (
    GaussianNormalizer,
    IdentityNormalizer,
    build_normalizer,
)
from pod_sim2real.model import PODBasis, fit_pod_bases_from_dataset
from pod_sim2real.training.trainer import evaluate_model


def test_gaussian_normalizer_round_trip():
    mean = [0.33, 0.01]
    std = [0.07, 0.03]
    normalizer = GaussianNormalizer(mean, std)

    # 5D tensor: [B, T, C, H, W]
    x = torch.randn(2, 4, 2, 8, 16)
    x_norm = normalizer.normalize(x)
    x_recon = normalizer.denormalize(x_norm)
    assert torch.allclose(x, x_recon, atol=1e-6)

    # Preprocess & postprocess
    y = torch.randn(2, 4, 2, 8, 16)
    x_p, y_p = normalizer.preprocess(x, y)
    assert torch.allclose(x_p, x_norm, atol=1e-6)
    x_d, y_d = normalizer.postprocess(x_p, y_p)
    assert torch.allclose(x, x_d, atol=1e-6)
    assert torch.allclose(y, y_d, atol=1e-6)


def test_gaussian_normalizer_shapes():
    mean = torch.tensor([1.0, 2.0])
    std = torch.tensor([0.5, 0.25])
    normalizer = GaussianNormalizer(mean, std)

    # 4D: [T, C, H, W]
    x4d = torch.ones(3, 2, 4, 4)
    x4d_norm = normalizer.normalize(x4d)
    assert torch.allclose(x4d_norm[:, 0], torch.zeros(3, 4, 4))
    assert torch.allclose(x4d_norm[:, 1], torch.full((3, 4, 4), -4.0))

    # 5D channels last: [B, T, H, W, C]
    x5d_last = torch.ones(2, 3, 4, 4, 2)
    x5d_norm = normalizer.normalize(x5d_last)
    assert torch.allclose(x5d_norm[..., 0], torch.zeros(2, 3, 4, 4))
    assert torch.allclose(x5d_norm[..., 1], torch.full((2, 3, 4, 4), -4.0))


def test_gaussian_normalizer_save_load(tmp_path):
    mean = [0.5, -0.5]
    std = [1.2, 0.8]
    norm = GaussianNormalizer(mean, std)
    save_path = tmp_path / "norm.pt"
    norm.save(save_path)

    loaded = GaussianNormalizer.from_file(save_path)
    assert torch.allclose(loaded.mean, norm.mean)
    assert torch.allclose(loaded.std, norm.std)


def test_gaussian_normalizer_from_dataset():
    # Mock dataset where channel 0 has mean 2.0, channel 1 has mean 4.0
    x = torch.zeros(10, 5, 2, 8, 8)
    x[:, :, 0] = 2.0
    x[:, :, 1] = 4.0
    y = torch.zeros(10, 5, 2, 8, 8)
    y[:, :, 0] = 2.0
    y[:, :, 1] = 4.0

    class MockDataset:
        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return x[idx], y[idx], {}

    norm = GaussianNormalizer.from_dataset(MockDataset(), max_samples=10)
    assert pytest.approx(norm.mean[0].item(), abs=1e-5) == 2.0
    assert pytest.approx(norm.mean[1].item(), abs=1e-5) == 4.0
    # Since constant, std should be clamped to at least eps
    assert norm.std[0].item() >= 1e-8


def test_evaluate_model_with_normalizer():
    class ShiftModel(torch.nn.Module):
        """Model that operates on normalized data and outputs input + 1 in normalized space."""
        def forward(self, x):
            return x + 1.0

    mean = [10.0, 20.0]
    std = [2.0, 4.0]
    normalizer = GaussianNormalizer(mean, std)

    # Physical data centered around mean
    x_phys = torch.zeros(4, 3, 2, 4, 4)
    x_phys[:, :, 0] = 10.0
    x_phys[:, :, 1] = 20.0

    # Target is in physical space: pred should be denormalize(0 + 1) = 1 * std + mean
    # So pred_phys[:, :, 0] = 12.0, pred_phys[:, :, 1] = 24.0
    y_phys = torch.zeros(4, 3, 2, 4, 4)
    y_phys[:, :, 0] = 12.0
    y_phys[:, :, 1] = 24.0

    ds = TensorDataset(x_phys, y_phys, torch.arange(4))
    res = evaluate_model(ShiftModel(), ds, device=torch.device("cpu"), batch_size=2, normalizer=normalizer)
    assert pytest.approx(res["mse"], abs=1e-6) == 0.0
    assert pytest.approx(res["rel_l2"], abs=1e-6) == 0.0

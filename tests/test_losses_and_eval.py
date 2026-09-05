import torch
import math
import pytest
from pod_sim2real.training.losses import compute_loss, compute_metrics, relative_tke_loss
from pod_sim2real.training.trainer import evaluate_model
from torch.utils.data import TensorDataset

def test_compute_loss_standard_mse():
    pred = torch.ones(2, 5, 2, 8, 8)
    target = torch.zeros(2, 5, 2, 8, 8)
    loss, parts = compute_loss(pred, target, kind='unet', use_tke=False)
    assert pytest.approx(loss.item()) == 1.0
    assert pytest.approx(parts['mse']) == 1.0
    assert pytest.approx(parts['tke']) == 0.0

def test_compute_loss_with_tke():
    pred = torch.ones(2, 5, 2, 8, 8) * 2.0
    target = torch.ones(2, 5, 2, 8, 8)
    loss, parts = compute_loss(pred, target, kind='unet', use_tke=True, tke_weight=0.1)
    assert parts['mse'] == 1.0
    assert parts['tke'] > 0.0
    assert loss.item() == pytest.approx(1.0 + 0.1 * parts['tke'])

def test_compute_loss_with_vorticity():
    pred = torch.ones(2, 5, 2, 8, 8) * 2.0
    target = torch.ones(2, 5, 2, 8, 8)
    loss, parts = compute_loss(pred, target, kind='triad-mno', use_vorticity=True, vorticity_weight=0.1)
    assert parts['mse'] == 1.0
    assert 'vorticity' in parts
    assert parts['vorticity'] >= 0.0
    assert loss.item() == pytest.approx(1.0 + 0.1 * parts['vorticity'])

def test_compute_metrics_exact():
    target = torch.full((1, 4, 2, 4, 4), 2.0)
    pred = torch.full((1, 4, 2, 4, 4), 3.0)
    metrics = compute_metrics(pred, target)
    assert pytest.approx(metrics['mse']) == 1.0
    assert pytest.approx(metrics['rmse']) == 1.0
    assert pytest.approx(metrics['mae']) == 1.0
    assert pytest.approx(metrics['rel_l2']) == 0.5
    assert pytest.approx(metrics['u_mse']) == 1.0
    assert pytest.approx(metrics['v_mse']) == 1.0
    assert pytest.approx(metrics['u_rel_l2']) == 0.5
    assert pytest.approx(metrics['v_rel_l2']) == 0.5

def test_evaluate_model_dictionary():
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x + 1.0
    x = torch.zeros(4, 3, 2, 4, 4)
    y = torch.zeros(4, 3, 2, 4, 4)
    ds = TensorDataset(x, y, torch.arange(4))
    res = evaluate_model(DummyModel(), ds, device=torch.device('cpu'), batch_size=2)
    assert 'mse' in res
    assert 'rmse' in res
    assert 'mae' in res
    assert 'rel_l2' in res
    assert 'u_mse' in res
    assert 'v_mse' in res
    assert 'vorticity_mse' in res
    assert 'vorticity_rel_l2' in res
    assert not math.isnan(res['vorticity_rel_l2'])
    assert pytest.approx(res['mse']) == 1.0

from pathlib import Path
import pytest
pytest.importorskip("pyarrow")
from pod_sim2real.data import discover_trajectories, pair_trajectories

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.skipif(not list((ROOT / "data/data_real").glob("*.arrow")), reason="smoke data not mounted")
def test_smoke_pairing_and_shapes():
    real = discover_trajectories(ROOT / "data/data_real", "real")
    sim = discover_trajectories(ROOT / "data/data_sim", "sim")
    pairs = pair_trajectories(real, sim)
    assert len(real) == len(sim) == 3
    assert all(pair.real and pair.sim for pair in pairs)
    assert real[0].shape == (3990, 128, 256)
    assert real[0].fields((32, 64)).shape == (3990, 2, 32, 64)

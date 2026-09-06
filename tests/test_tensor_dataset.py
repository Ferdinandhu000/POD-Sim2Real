import json
from pathlib import Path
import numpy as np
import pytest
import torch

datasets = pytest.importorskip("datasets")

from pod_sim2real.data import (
    OfficialArrowWindowDataset,
    PrecomputedTrajectoryDataset,
    preprocess_domain,
    build_trajectory_prefix_entries,
)


def _write_trajectory_dataset(directory, sim_ids, frames=10, height=8, width=16):
    u, v = [], []
    for value in range(len(sim_ids)):
        field = (np.arange(frames * height * width, dtype=np.float32) + value * 1000).reshape(frames, height, width)
        u.append(field.tobytes())
        v.append((field + 100).tobytes())
    datasets.Dataset.from_dict({
        "sim_id": sim_ids,
        "shape_t": [frames] * len(sim_ids),
        "shape_h": [height] * len(sim_ids),
        "shape_w": [width] * len(sim_ids),
        "u": u,
        "v": v,
    }).save_to_disk(str(directory))


def test_preprocess_and_precomputed_dataset(tmp_path):
    raw_dir = tmp_path / "raw"
    tensor_dir = tmp_path / "tensor_cache"
    sim_ids = ["traj_0.h5", "traj_1.h5"]
    _write_trajectory_dataset(raw_dir, sim_ids, frames=10, height=8, width=16)

    target_res = (4, 8)
    # Preprocess
    preprocess_domain(
        input_dir=raw_dir,
        output_dir=tensor_dir,
        target_res=target_res,
        prefix_frames=None,
        overwrite=True,
        num_workers=1,
    )

    # Check files created
    assert (tensor_dir / "traj_0.h5.pt").exists()
    assert (tensor_dir / "traj_1.h5.pt").exists()

    pt_data = torch.load(tensor_dir / "traj_0.h5.pt", map_location="cpu", weights_only=True)
    assert pt_data.shape == (10, 2, 4, 8)
    assert pt_data.dtype == torch.float32

    # Now create entries
    index_entries = [
        {"sim_id": "traj_0.h5", "time_id": 0},
        {"sim_id": "traj_0.h5", "time_id": 2},
        {"sim_id": "traj_1.h5", "time_id": 1},
    ]

    # Compare ArrowWindowDataset with PrecomputedTrajectoryDataset
    arrow_ds = OfficialArrowWindowDataset(
        raw_dir,
        None,
        input_steps=2,
        output_steps=2,
        resolution=target_res,
        index_entries=index_entries,
    )

    tensor_ds = PrecomputedTrajectoryDataset(
        tensor_dir=tensor_dir,
        index_file=None,
        index_entries=index_entries,
        input_steps=2,
        output_steps=2,
        resolution=target_res,
        in_memory=True,
    )

    assert len(arrow_ds) == len(tensor_ds) == 3

    for i in range(len(arrow_ds)):
        x_arrow, y_arrow, idx_arrow = arrow_ds[i]
        x_tensor, y_tensor, idx_tensor = tensor_ds[i]
        assert idx_arrow == idx_tensor == i
        torch.testing.assert_close(x_arrow, x_tensor, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(y_arrow, y_tensor, atol=1e-5, rtol=1e-5)

    # Test in_memory=False
    tensor_ds_disk = PrecomputedTrajectoryDataset(
        tensor_dir=tensor_dir,
        index_file=None,
        index_entries=index_entries,
        input_steps=2,
        output_steps=2,
        resolution=target_res,
        in_memory=False,
    )
    x_disk, y_disk, _ = tensor_ds_disk[0]
    torch.testing.assert_close(arrow_ds[0][0], x_disk, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(arrow_ds[0][1], y_disk, atol=1e-5, rtol=1e-5)

    # Test prefix_frames dynamic slicing in RAM
    tensor_ds_prefix = PrecomputedTrajectoryDataset(
        tensor_dir=tensor_dir,
        index_file=None,
        index_entries=[{"sim_id": "traj_0.h5", "time_id": 0}],
        input_steps=2,
        output_steps=2,
        resolution=target_res,
        prefix_frames=0.5,
        in_memory=True,
    )
    assert tensor_ds_prefix.trajectories["traj_0.h5"].shape[0] == 5
    x_pref, y_pref, _ = tensor_ds_prefix[0]
    torch.testing.assert_close(arrow_ds[0][0], x_pref, atol=1e-5, rtol=1e-5)



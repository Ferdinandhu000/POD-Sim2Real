import json

import numpy as np
import pytest

datasets = pytest.importorskip("datasets")

from pod_sim2real.data import (
    OfficialArrowWindowDataset,
    build_trajectory_prefix_entries,
)


def _write_trajectory_dataset(directory, sim_ids):
    frames, height, width = 6, 4, 8
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


def test_official_window_dataset_slices_and_filters(tmp_path):
    foil = tmp_path / "foil"
    index_root = foil / "hf_dataset"
    real_dir = index_root / "real"
    _write_trajectory_dataset(real_dir, ["10000_0.0.h5", "20000_10.0.h5"])
    entries = [
        {"sim_id": "10000_0.0.h5", "time_id": 1},
        {"sim_id": "20000_10.0.h5", "time_id": 2},
    ]
    (index_root / "test_index_real.json").write_text(json.dumps(entries), encoding="utf-8")
    (foil / "in_dist_test_params_real.json").write_text(json.dumps({"20000_10.0.h5": {}}), encoding="utf-8")

    dataset = OfficialArrowWindowDataset(
        real_dir,
        index_root / "test_index_real.json",
        input_steps=2,
        output_steps=2,
        resolution=(2, 4),
        test_mode="in_dist",
        metadata_root=foil,
    )

    assert len(dataset) == 1
    inputs, outputs, sample_index = dataset[0]
    assert sample_index == 0
    assert inputs.shape == (2, 2, 2, 4)
    assert outputs.shape == (2, 2, 2, 4)
    # The selected trajectory begins at 1000 and the selected window at t=2.
    assert inputs[0, 0, 0, 0].item() == pytest.approx(1064.0)
    assert inputs[0, 1, 0, 0].item() == pytest.approx(1164.0)


def test_trajectory_prefix_never_reuses_a_trajectory(tmp_path):
    foil = tmp_path / "foil"
    real_dir = foil / "hf_dataset" / "real"
    frames, height, width = 120, 4, 8
    field = np.zeros((frames, height, width), dtype=np.float32)
    sim_ids = ["10000_0.0.h5", "10000_5.0.h5", "20000_0.0.h5", "20000_5.0.h5"]
    datasets.Dataset.from_dict({
        "sim_id": sim_ids,
        "shape_t": [frames] * len(sim_ids),
        "shape_h": [height] * len(sim_ids),
        "shape_w": [width] * len(sim_ids),
        "u": [field.tobytes()] * len(sim_ids),
        "v": [field.tobytes()] * len(sim_ids),
    }).save_to_disk(str(real_dir))

    split_ids = {
        "train": sim_ids[:2],
        "val": sim_ids[2:3],
        "test": sim_ids[3:],
    }
    entries = build_trajectory_prefix_entries(
        real_dir, 20, 20, 20, prefix_frames=60,
        trajectory_splits=split_ids,
    )
    ids_by_split = {split: {item["sim_id"] for item in values} for split, values in entries.items()}
    assert ids_by_split == {"train": set(sim_ids[:2]), "val": {sim_ids[2]}, "test": {sim_ids[3]}}
    assert not (ids_by_split["train"] & ids_by_split["val"])
    assert not (ids_by_split["train"] & ids_by_split["test"])
    assert not (ids_by_split["val"] & ids_by_split["test"])
    # With frames=120, prefix_frames=60, horizon=40, stride=20: starts should be 0, 20 (max_start = 60-40 = 20)
    for values in entries.values():
        for item in values:
            assert item["time_id"] <= 20


def test_trajectory_prefix_rejects_duplicate_split_ids(tmp_path):
    real_dir = tmp_path / "real"
    frames, height, width = 120, 4, 8
    field = np.zeros((frames, height, width), dtype=np.float32)
    sim_ids = ["10000_0.0.h5", "10000_5.0.h5", "20000_0.0.h5"]
    datasets.Dataset.from_dict({
        "sim_id": sim_ids,
        "shape_t": [frames] * len(sim_ids),
        "shape_h": [height] * len(sim_ids),
        "shape_w": [width] * len(sim_ids),
        "u": [field.tobytes()] * len(sim_ids),
        "v": [field.tobytes()] * len(sim_ids),
    }).save_to_disk(str(real_dir))

    with pytest.raises(ValueError, match="appears in more than one split"):
        build_trajectory_prefix_entries(
            real_dir, 20, 20, 20,
            trajectory_splits={"train": [sim_ids[0]], "val": [sim_ids[0]], "test": [sim_ids[1]]},
        )

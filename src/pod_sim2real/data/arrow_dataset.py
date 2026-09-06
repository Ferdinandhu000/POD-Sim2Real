from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from torch.utils.data import Dataset


def _dtype(raw: bytes, count: int) -> np.dtype:
    if count and len(raw) == count * 8:
        return np.dtype("<f8")
    if count and len(raw) == count * 4:
        return np.dtype("<f4")
    raise ValueError(f"cannot infer numeric dtype: {len(raw)} bytes for {count} values")


def decode_array(raw: bytes | None, shape: tuple[int, ...], *, name: str = "array") -> np.ndarray | None:
    if raw is None:
        return None
    count = int(np.prod(shape))
    dtype = _dtype(raw, count)
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def setting_from_id(sim_id: str) -> tuple[int, float]:
    match = re.match(r"^(?P<re>[-+]?\d+(?:\.\d+)?)_(?P<aoa>[-+]?\d+(?:\.\d+)?)", Path(sim_id).stem)
    if not match:
        raise ValueError(f"expected Re_AoA identifier, got {sim_id!r}")
    return int(float(match.group("re"))), float(match.group("aoa"))


@dataclass
class ArrowTrajectory:
    domain: str
    sim_id: str
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray | None
    t: np.ndarray
    x: np.ndarray | None
    y: np.ndarray | None
    source: str
    _field_cache: dict[tuple[int, int], np.ndarray] | None = None

    @property
    def setting(self) -> tuple[int, float]:
        return setting_from_id(self.sim_id)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.u.shape)  # type: ignore[return-value]

    def fields(self, resolution: tuple[int, int] | None = None) -> np.ndarray:
        if self._field_cache is None:
            self._field_cache = {}
        key = resolution or (self.u.shape[1], self.u.shape[2])
        if key in self._field_cache:
            return self._field_cache[key]
        arrays = [self.u, self.v]
        if resolution is not None and arrays[0].shape[1:] != resolution:
            sh, sw = arrays[0].shape[1] // resolution[0], arrays[0].shape[2] // resolution[1]
            if arrays[0].shape[1] % resolution[0] or arrays[0].shape[2] % resolution[1]:
                raise ValueError(f"resolution {resolution} is not an integer subsample of {arrays[0].shape[1:]}")
            arrays = [a[:, ::sh, ::sw] for a in arrays]
        result = np.stack(arrays, axis=1).astype(np.float32, copy=False)
        self._field_cache[key] = result
        return result


def _read_batch(path: Path) -> Iterable[dict]:
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_stream(source)
        for batch in reader:
            data = batch.to_pydict()
            for i in range(batch.num_rows):
                yield {key: values[i] for key, values in data.items()}


def read_trajectory(path: str | Path, domain: str, record_index: int = 0) -> ArrowTrajectory:
    path = Path(path)
    rows = list(_read_batch(path))
    row = rows[record_index]
    shape = (int(row["shape_t"]), int(row["shape_h"]), int(row["shape_w"]))
    sim_id = str(row["sim_id"])
    return ArrowTrajectory(domain, sim_id, decode_array(row["u"], shape, name="u"), decode_array(row["v"], shape, name="v"), decode_array(row.get("p"), shape, name="p"), decode_array(row["t"], (int(row["t_shape"]),), name="t"), decode_array(row.get("x"), (int(row["x_shape_h"]), int(row["x_shape_w"])), name="x"), decode_array(row.get("y"), (int(row["y_shape_h"]), int(row["y_shape_w"])), name="y"), str(path))


def discover_trajectories(root: str | Path, domain: str) -> list[ArrowTrajectory]:
    directory = Path(root)
    paths = sorted(directory.glob("*.arrow"))
    if not paths:
        raise FileNotFoundError(f"no Arrow files in {directory}")
    records: list[ArrowTrajectory] = []
    for path in paths:
        for index, row in enumerate(_read_batch(path)):
            shape = (int(row["shape_t"]), int(row["shape_h"]), int(row["shape_w"]))
            records.append(ArrowTrajectory(domain, str(row["sim_id"]), decode_array(row["u"], shape, name="u"), decode_array(row["v"], shape, name="v"), decode_array(row.get("p"), shape, name="p"), decode_array(row["t"], (int(row["t_shape"]),), name="t"), decode_array(row.get("x"), (int(row["x_shape_h"]), int(row["x_shape_w"])), name="x"), decode_array(row.get("y"), (int(row["y_shape_h"]), int(row["y_shape_w"])), name="y"), f"{path}#record={index}"))
    return records


@dataclass(frozen=True)
class WindowRef:
    trajectory: ArrowTrajectory
    start: int


class ArrowWindowDataset(Dataset):
    def __init__(self, trajectories: list[ArrowTrajectory], input_steps: int = 20, output_steps: int = 20, stride: int = 20, resolution: tuple[int, int] = (32, 64), index_entries: list[dict[str, int | str]] | None = None):
        self.trajectories, self.input_steps, self.output_steps, self.stride, self.resolution = trajectories, input_steps, output_steps, stride, resolution
        if index_entries is None:
            self.refs = [WindowRef(t, start) for t in trajectories for start in range(0, t.u.shape[0] - input_steps - output_steps + 1, stride)]
        else:
            by_id = {t.sim_id: t for t in trajectories}
            self.refs = []
            for item in index_entries:
                trajectory = by_id.get(str(item["sim_id"]))
                start = int(item["time_id"])
                if trajectory is None:
                    continue
                if 0 <= start <= trajectory.u.shape[0] - input_steps - output_steps:
                    self.refs.append(WindowRef(trajectory, start))
        if not self.refs:
            raise ValueError("no valid windows")

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        ref = self.refs[index]
        fields = ref.trajectory.fields(self.resolution)
        split = self.input_steps
        x = torch.from_numpy(fields[ref.start:ref.start + split])
        y = torch.from_numpy(fields[ref.start + split:ref.start + split + self.output_steps])
        return x, y, self.trajectories.index(ref.trajectory)


class OfficialArrowWindowDataset(Dataset):
    """Lazy window dataset for the official RealPDEBench V2 Arrow layout.

    Hugging Face Arrow rows contain one complete trajectory. The JSON index
    supplies only ``sim_id`` and ``time_id``; binary fields are decoded and
    sliced when a sample is requested, so all trajectories are never resident
    in RAM at once.
    """
    def __init__(
        self,
        arrow_dir: str | Path,
        index_file: str | Path | None,
        input_steps: int = 20,
        output_steps: int = 20,
        resolution: tuple[int, int] = (32, 64),
        *,
        test_mode: str = "all",
        metadata_root: str | Path | None = None,
        index_entries: list[dict[str, int | str]] | None = None,
        prefix_frames: int | None = None,
        cache_trajectories: bool = True,
    ):
        from datasets import load_from_disk
        self.arrow_dir = Path(arrow_dir)
        self.index_file = Path(index_file) if index_file is not None else None
        self.input_steps, self.output_steps, self.resolution = input_steps, output_steps, resolution
        self.table = load_from_disk(str(self.arrow_dir), keep_in_memory=False)
        if index_entries is not None:
            self.entries = index_entries
        elif self.index_file is not None:
            self.entries = json.loads(self.index_file.read_text(encoding="utf-8"))
        else:
            raise ValueError("index_file or index_entries is required")
        if test_mode != "all":
            if test_mode not in {"seen", "in_dist", "out_dist", "unseen"}:
                raise ValueError(f"unsupported test_mode={test_mode!r}")
            metadata_dir = Path(metadata_root) if metadata_root else self.arrow_dir.parent.parent
            domain = "real" if self.arrow_dir.name == "real" else "numerical"
            def load_ids(name: str) -> set[str]:
                path = metadata_dir / f"{name}_params_{domain}.json"
                if not path.exists():
                    raise FileNotFoundError(f"missing official test metadata: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                return set(payload.keys())
            if test_mode == "seen":
                accepted_ids = load_ids("remain")
            elif test_mode == "in_dist":
                accepted_ids = load_ids("in_dist_test")
            elif test_mode == "out_dist":
                accepted_ids = load_ids("out_dist_test")
            else:
                accepted_ids = load_ids("in_dist_test") | load_ids("out_dist_test")
            self.entries = [entry for entry in self.entries if str(entry["sim_id"]) in accepted_ids]
        # Selecting only metadata columns avoids touching multi-gigabyte binary columns
        # while constructing the lookup table.
        meta_table = self.table.select_columns(["sim_id", "shape_t", "shape_h", "shape_w"])
        self.row_by_id = {str(meta_table[i]["sim_id"]): i for i in range(len(meta_table))}
        self.shapes_by_row = [
            (int(meta_table[i]["shape_t"]), int(meta_table[i]["shape_h"]), int(meta_table[i]["shape_w"]))
            for i in range(len(meta_table))
        ]
        self.valid_entries = [e for e in self.entries if str(e["sim_id"]) in self.row_by_id]
        if prefix_frames is not None and int(prefix_frames) > 0:
            horizon = self.input_steps + self.output_steps
            max_start = int(prefix_frames) - horizon
            self.valid_entries = [e for e in self.valid_entries if int(e["time_id"]) <= max_start]
        if not self.valid_entries:
            raise ValueError(f"no indexed trajectories found in {self.arrow_dir} for {self.index_file}")

        # PyArrow table reference for zero-copy memory-mapped slice (5600x faster than full row deserialization)
        pa_data = getattr(self.table, "_data", None)
        self._pa_table = getattr(pa_data, "table", pa_data)
        self._has_arrow_buffers = (
            self._pa_table is not None
            and hasattr(self._pa_table, "__getitem__")
            and hasattr(self._pa_table, "column_names")
            and "u" in self._pa_table.column_names
            and "v" in self._pa_table.column_names
        )

        # In-memory trajectory cache: caches downsampled (T, res_h, res_w) arrays in RAM.
        # Eliminates repeated disk I/O and GB-scale deserialization for thousands of sliding windows.
        self.cache_trajectories = cache_trajectories
        self._traj_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if self.cache_trajectories:
            unique_rows = {self.row_by_id[str(e["sim_id"])] for e in self.valid_entries}
            if len(unique_rows) <= 128:
                for r_idx in unique_rows:
                    self._get_trajectory(r_idx)

    @staticmethod
    def _decode(raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
        count = int(np.prod(shape))
        dtype = np.float64 if len(raw) == count * 8 else np.float32
        arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
        if dtype == np.float64:
            arr = arr.astype(np.float32, copy=False)
        return arr

    def _slice_field(
        self, channel_name: str, row_idx: int, full_shape: tuple[int, int, int], start: int, stop: int, sh: int, sw: int
    ) -> np.ndarray:
        if self._has_arrow_buffers:
            try:
                col = self._pa_table[channel_name]
                scalar = col[row_idx]
                if hasattr(scalar, "as_buffer"):
                    buf = scalar.as_buffer()
                    total_count = full_shape[0] * full_shape[1] * full_shape[2]
                    itemsize = 8 if buf.size == total_count * 8 else 4
                    dtype = np.float64 if itemsize == 8 else np.float32
                    frame_bytes = full_shape[1] * full_shape[2] * itemsize
                    start_byte = start * frame_bytes
                    length_bytes = (stop - start) * frame_bytes
                    sub_buf = buf.slice(start_byte, length_bytes)
                    arr = np.frombuffer(sub_buf, dtype=dtype).reshape(stop - start, full_shape[1], full_shape[2])
                    if sh != 1 or sw != 1:
                        arr = arr[:, ::sh, ::sw]
                    if itemsize == 8:
                        arr = arr.astype(np.float32, copy=False)
                    return arr
            except Exception as e:
                if not getattr(self, "_warned_fallback", False):
                    import logging
                    logging.getLogger(__name__).warning(
                        "Arrow buffer zero-copy slicing failed (%s: %s). Falling back to full-row deserialization.",
                        type(e).__name__, e,
                    )
                    self._warned_fallback = True
        # Fallback to standard row access if arrow buffer slicing is unavailable
        row = self.table[row_idx]
        raw = row[channel_name]
        return self._decode(raw, full_shape)[start:stop, ::sh, ::sw]

    def _get_trajectory(self, row_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if row_idx in self._traj_cache:
            return self._traj_cache[row_idx]
        full_shape = self.shapes_by_row[row_idx]
        sh = full_shape[1] // int(self.resolution[0])
        sw = full_shape[2] // int(self.resolution[1])
        u = self._slice_field("u", row_idx, full_shape, 0, full_shape[0], sh, sw)
        v = self._slice_field("v", row_idx, full_shape, 0, full_shape[0], sh, sw)
        self._traj_cache[row_idx] = (u, v)
        return u, v

    def __len__(self) -> int:
        return len(self.valid_entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        entry = self.valid_entries[index]
        row_idx = self.row_by_id[str(entry["sim_id"])]
        full_shape = self.shapes_by_row[row_idx]
        start = int(entry["time_id"])
        stop = start + self.input_steps + self.output_steps
        if start < 0 or stop > full_shape[0]:
            raise IndexError(f"window [{start}, {stop}) exceeds trajectory length {full_shape[0]}")
        sh = full_shape[1] // int(self.resolution[0])
        sw = full_shape[2] // int(self.resolution[1])
        if full_shape[1] % int(self.resolution[0]) or full_shape[2] % int(self.resolution[1]):
            raise ValueError(f"resolution {self.resolution} is not an integer subsample of {full_shape[1:]}")
        if self.cache_trajectories:
            u_full, v_full = self._get_trajectory(row_idx)
            u = u_full[start:stop]
            v = v_full[start:stop]
        else:
            u = self._slice_field("u", row_idx, full_shape, start, stop, sh, sw)
            v = self._slice_field("v", row_idx, full_shape, start, stop, sh, sw)
        fields = torch.from_numpy(np.stack((u, v), axis=1).astype(np.float32, copy=False))
        return fields[:self.input_steps], fields[self.input_steps:], index


def _trajectory_rows(arrow_dir: str | Path) -> list[dict[str, int | str]]:
    """Read only trajectory identifiers and lengths from a HF dataset."""
    from datasets import load_from_disk

    table = load_from_disk(str(arrow_dir), keep_in_memory=False).select_columns(["sim_id", "shape_t"])
    return [{"sim_id": str(row["sim_id"]), "shape_t": int(row["shape_t"])} for row in table]


def _load_index_trajectory_ids(index_root: str | Path, domain: str) -> dict[str, list[str]]:
    """Return unique trajectory IDs assigned to official train/val/test indices."""
    root = Path(index_root)
    suffix = "real" if domain == "real" else "numerical"
    paths = {split: root / f"{split}_index_{suffix}.json" for split in ("train", "val", "test")}
    if not all(path.exists() for path in paths.values()):
        return {}
    result: dict[str, list[str]] = {}
    for split, path in paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[split] = sorted({str(item["sim_id"]) for item in payload})
    return result


def split_trajectory_ids(
    arrow_dir: str | Path,
    *,
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, list[str]]:
    """Deterministically split complete trajectory files by filename."""
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("invalid trajectory split fractions")
    rows = _trajectory_rows(arrow_dir)
    ids = sorted(str(row["sim_id"]) for row in rows)
    if len(ids) < 3:
        raise ValueError("at least three trajectories are required for train/val/test")
    import random

    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = max(1, int(n * train_fraction))
    n_val = max(1, int(n * val_fraction))
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    return {"train": sorted(ids[:n_train]), "val": sorted(ids[n_train:n_train + n_val]), "test": sorted(ids[n_train + n_val:])}


def build_trajectory_prefix_entries(
    arrow_dir: str | Path,
    input_steps: int,
    output_steps: int,
    stride: int,
    *,
    prefix_frames: int | None = None,
    trajectory_splits: dict[str, list[str]] | None = None,
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, list[dict[str, int | str]]]:
    """Build windows from trajectories assigned to each split.
    If prefix_frames is specified and > 0, cuts to prefix_frames; otherwise uses the full trajectory.
    """
    rows = _trajectory_rows(arrow_dir)
    by_id = {str(row["sim_id"]): int(row["shape_t"]) for row in rows}
    splits = trajectory_splits or split_trajectory_ids(arrow_dir, seed=seed, train_fraction=train_fraction, val_fraction=val_fraction)
    seen: set[str] = set()
    result = {"train": [], "val": [], "test": []}
    horizon = int(input_steps) + int(output_steps)
    for split in ("train", "val", "test"):
        for sim_id in splits.get(split, []):
            sim_id = str(sim_id)
            if sim_id in seen:
                raise ValueError(f"trajectory {sim_id!r} appears in more than one split")
            seen.add(sim_id)
            if sim_id not in by_id:
                continue
            if prefix_frames is not None and int(prefix_frames) > 0:
                prefix_end = min(by_id[sim_id], max(horizon, int(prefix_frames)))
            else:
                prefix_end = by_id[sim_id]
            starts = range(0, prefix_end - horizon + 1, max(1, int(stride)))
            result[split].extend({"sim_id": sim_id, "time_id": int(time_id)} for time_id in starts)
    if any(not result[split] for split in result):
        raise ValueError(f"trajectory prefix split produced an empty split: { {k: len(v) for k, v in result.items()} }")
    return result

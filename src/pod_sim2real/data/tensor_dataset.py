from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class PrecomputedTrajectoryDataset(Dataset):
    """Ultra-fast, zero-overhead Dataset that loads pre-downsampled trajectory tensors (.pt).

    Features:
    1. In-Memory Shared Tensors (in_memory=True):
       All trajectories for the current split are loaded once into RAM at startup
       and registered with .share_memory_(). PyTorch DataLoader workers access
       the tensors with ZERO disk I/O, ZERO IPC duplication, and ZERO deserialization overhead.
    2. Zero-Copy Slicing:
       __getitem__ performs pure PyTorch pointer slicing (window = traj[start : start + horizon]),
       taking microseconds per batch.
    3. Seamless compatibility with OfficialArrowWindowDataset:
       Handles test_mode, prefix_frames, official index files, and trajectory prefix splits.
    """

    def __init__(
        self,
        tensor_dir: str | Path,
        index_file: str | Path | None = None,
        input_steps: int = 20,
        output_steps: int = 20,
        resolution: tuple[int, int] = (64, 128),
        *,
        test_mode: str = "all",
        metadata_root: str | Path | None = None,
        index_entries: list[dict[str, int | str]] | None = None,
        prefix_frames: int | float | None = None,
        in_memory: bool = True,
        mmap: bool = False,
    ):
        super().__init__()
        self.tensor_dir = Path(tensor_dir)
        if not self.tensor_dir.exists():
            raise FileNotFoundError(f"tensor_dir not found: {self.tensor_dir}")

        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.horizon = self.input_steps + self.output_steps
        self.resolution = tuple(resolution)
        self.in_memory = in_memory
        self.mmap = mmap

        # 1. Parse entries
        if index_entries is not None:
            self.entries = [dict(e) for e in index_entries]
        elif index_file is not None:
            self.index_file = Path(index_file)
            self.entries = json.loads(self.index_file.read_text(encoding="utf-8"))
        else:
            raise ValueError("Either index_file or index_entries must be provided")

        # 2. Filter by test_mode (official test subsets)
        if test_mode != "all":
            if test_mode not in {"seen", "in_dist", "out_dist", "unseen"}:
                raise ValueError(f"unsupported test_mode={test_mode!r}")
            meta_dir = Path(metadata_root) if metadata_root else self.tensor_dir.parent.parent
            domain = "real" if "real" in self.tensor_dir.name else "numerical"

            def load_ids(name: str) -> set[str]:
                path = meta_dir / f"{name}_params_{domain}.json"
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

        # 3. Discover trajectory files
        # Files are named e.g. "10000_0.0.h5.pt" or "10000_0.0.pt"
        self.file_by_id: dict[str, Path] = {}
        for p in self.tensor_dir.glob("*.pt"):
            stem = p.name[:-3] if p.name.endswith(".pt") else p.stem
            self.file_by_id[stem] = p
            # Also register stem without .h5 if present
            if stem.endswith(".h5"):
                self.file_by_id[stem[:-3]] = p
            else:
                self.file_by_id[f"{stem}.h5"] = p

        # Identify which trajectories are actually referenced in this split
        needed_ids = {str(e["sim_id"]) for e in self.entries if str(e["sim_id"]) in self.file_by_id}

        # 4. Load trajectories
        self.trajectories: dict[str, torch.Tensor] = {}
        self.shapes_by_id: dict[str, tuple[int, ...]] = {}

        for sim_id in sorted(needed_ids):
            pt_path = self.file_by_id[sim_id]
            if self.in_memory:
                # Load fully into CPU RAM
                tensor = torch.load(pt_path, weights_only=True, map_location="cpu")
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.from_numpy(np.asarray(tensor))
                tensor = tensor.float().contiguous()
                # Apply prefix_frames slice if needed to reduce RAM usage
                if prefix_frames is not None:
                    total_t = tensor.shape[0]
                    prefix_val = float(prefix_frames)
                    if 0.0 < prefix_val <= 1.0:
                        max_frames = int(total_t * prefix_val)
                    else:
                        max_frames = min(total_t, int(prefix_val))
                    tensor = tensor[:max_frames].contiguous()
                # Enable shared memory across DataLoader workers
                tensor.share_memory_()
                self.trajectories[sim_id] = tensor
                self.shapes_by_id[sim_id] = tuple(tensor.shape)
            else:
                # Read metadata/shape only for lazy loading
                if hasattr(torch, "load") and self.mmap:
                    try:
                        tensor = torch.load(pt_path, weights_only=True, map_location="cpu", mmap=True)
                        self.trajectories[sim_id] = tensor
                        self.shapes_by_id[sim_id] = tuple(tensor.shape)
                    except Exception:
                        tensor = torch.load(pt_path, weights_only=True, map_location="cpu")
                        self.shapes_by_id[sim_id] = tuple(tensor.shape)
                else:
                    # Fallback
                    tensor = torch.load(pt_path, weights_only=True, map_location="cpu")
                    self.shapes_by_id[sim_id] = tuple(tensor.shape)

        # 5. Filter valid entries based on horizon and prefix bounds
        self.valid_entries: list[dict[str, int | str]] = []
        for e in self.entries:
            sim_id = str(e["sim_id"])
            if sim_id not in self.shapes_by_id:
                continue
            total_t = self.shapes_by_id[sim_id][0]
            start = int(e["time_id"])
            max_start = total_t - self.horizon
            if 0 <= start <= max_start:
                self.valid_entries.append(e)

        if not self.valid_entries:
            raise ValueError(f"no valid indexed windows found in {self.tensor_dir}")

    def __len__(self) -> int:
        return len(self.valid_entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        entry = self.valid_entries[index]
        sim_id = str(entry["sim_id"])
        start = int(entry["time_id"])
        stop = start + self.horizon

        if self.in_memory or sim_id in self.trajectories:
            traj = self.trajectories[sim_id]
        else:
            # Lazy load from disk
            pt_path = self.file_by_id[sim_id]
            traj = torch.load(pt_path, weights_only=True, map_location="cpu").float()

        window = traj[start:stop]
        return window[:self.input_steps], window[self.input_steps:], index

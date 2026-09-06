from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
from typing import Tuple

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from tqdm import tqdm


def process_single_arrow(
    arrow_path: Path,
    out_dir: Path,
    target_res: tuple[int, int],
    prefix_frames: int | float | None = None,
    overwrite: bool = False,
) -> list[dict]:
    """Read an Arrow trajectory file, downsample to target_res, and save as float32 .pt."""
    results = []
    with pa.memory_map(str(arrow_path), "r") as src:
        reader = ipc.open_stream(src)
        for batch in reader:
            for row_idx in range(batch.num_rows):
                sim_id = str(batch.column("sim_id")[row_idx].as_py())
                shape_t = int(batch.column("shape_t")[row_idx].as_py())
                shape_h = int(batch.column("shape_h")[row_idx].as_py())
                shape_w = int(batch.column("shape_w")[row_idx].as_py())

                target_file = out_dir / f"{sim_id}.pt"
                if target_file.exists() and not overwrite:
                    results.append({
                        "sim_id": sim_id,
                        "status": "skipped",
                        "path": str(target_file),
                        "shape": [shape_t, 2, target_res[0], target_res[1]],
                    })
                    continue

                u_buf = batch.column("u")[row_idx].as_buffer()
                v_buf = batch.column("v")[row_idx].as_buffer()

                dtype = np.float32 if u_buf.size == shape_t * shape_h * shape_w * 4 else np.float64
                u = np.frombuffer(u_buf, dtype=dtype).reshape(shape_t, shape_h, shape_w)
                v = np.frombuffer(v_buf, dtype=dtype).reshape(shape_t, shape_h, shape_w)

                sh = shape_h // target_res[0]
                sw = shape_w // target_res[1]
                if shape_h % target_res[0] != 0 or shape_w % target_res[1] != 0:
                    raise ValueError(f"Target resolution {target_res} must evenly divide raw shape ({shape_h}, {shape_w})")

                u_down = u[:, ::sh, ::sw].astype(np.float32, copy=False)
                v_down = v[:, ::sh, ::sw].astype(np.float32, copy=False)

                if prefix_frames is not None:
                    pval = float(prefix_frames)
                    max_t = int(shape_t * pval) if 0.0 < pval <= 1.0 else min(shape_t, int(pval))
                    u_down = u_down[:max_t]
                    v_down = v_down[:max_t]

                stacked = np.stack([u_down, v_down], axis=1)  # (T, 2, H, W)
                tensor = torch.from_numpy(stacked).contiguous()

                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(tensor, target_file)

                results.append({
                    "sim_id": sim_id,
                    "status": "converted",
                    "path": str(target_file),
                    "shape": list(tensor.shape),
                    "bytes": target_file.stat().st_size,
                })
    return results


def _worker_wrapper(args):
    return process_single_arrow(*args)


def preprocess_domain(
    input_dir: Path,
    output_dir: Path,
    target_res: tuple[int, int],
    prefix_frames: int | float | None = None,
    overwrite: bool = False,
    num_workers: int = 4,
    desc: str = "Processing",
) -> dict[str, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrow_files = sorted(input_dir.glob("*.arrow"))
    if not arrow_files:
        return {}

    tasks = [
        (f, output_dir, target_res, prefix_frames, overwrite)
        for f in arrow_files
    ]

    results = {}
    if num_workers <= 1:
        for t in tqdm(tasks, desc=desc):
            res_list = _worker_wrapper(t)
            for res in res_list:
                results[res["sim_id"]] = res
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker_wrapper, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(tasks), desc=desc):
                for res in fut.result():
                    results[res["sim_id"]] = res

    return results

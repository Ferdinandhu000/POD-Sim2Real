#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from pod_sim2real.data.preprocess import preprocess_domain


def main():
    parser = argparse.ArgumentParser(description="Preprocess RealPDEBench Arrow files into fast downsampled Tensor (.pt) files")
    parser.add_argument("--data-root", type=Path, default=Path("data/foil"), help="Root path of foil dataset (e.g. data/foil)")
    parser.add_argument("--real-dir", type=Path, default=None, help="Explicit path to real arrow files directory")
    parser.add_argument("--sim-dir", type=Path, default=None, help="Explicit path to numerical/sim arrow files directory")
    parser.add_argument("--resolution", nargs=2, type=int, default=[64, 128], help="Target downsampled spatial resolution [H, W]")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output root directory for cached tensors")
    parser.add_argument("--prefix-frames", type=float, default=None, help="Optional frame cutoff (default None: stores all frames)")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing .pt files")
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Number of worker processes")
    args = parser.parse_args()

    res = tuple(args.resolution)

    # 1. Resolve source directories
    real_dir = args.real_dir
    sim_dir = args.sim_dir
    if real_dir is None or sim_dir is None:
        candidates = [
            (args.data_root / "hf_dataset" / "real", args.data_root / "hf_dataset" / "numerical"),
            (args.data_root / "real", args.data_root / "numerical"),
            (Path("data/data_real"), Path("data/data_sim")),
        ]
        for r_cand, s_cand in candidates:
            if r_cand.exists() and s_cand.exists():
                real_dir = real_dir or r_cand
                sim_dir = sim_dir or s_cand
                break

    if real_dir is None or not real_dir.exists():
        raise FileNotFoundError(f"Real data directory not found. Please specify --real-dir.")
    if sim_dir is None or not sim_dir.exists():
        raise FileNotFoundError(f"Sim data directory not found. Please specify --sim-dir.")

    # 2. Resolve output directory
    if args.output_dir is not None:
        out_root = args.output_dir
    else:
        # Default layout: data/foil/tensor_cache_64x128 or data/tensor_cache_64x128
        parent = real_dir.parent.parent if real_dir.parent.name in ("hf_dataset", "real") else real_dir.parent
        out_root = parent / f"tensor_cache_{res[0]}x{res[1]}"

    real_out = out_root / "real"
    sim_out = out_root / "numerical"

    print("=" * 80)
    print("Precomputing Downsampled Trajectory Tensors for Fast In-Memory Training")
    print("=" * 80)
    print(f"Target Resolution:  {res[0]} x {res[1]}")
    print(f"Real Source:        {real_dir}")
    print(f"Sim Source:         {sim_dir}")
    print(f"Output Directory:   {out_root}")
    print(f"Prefix Frames:      {args.prefix_frames or 'Full (All frames)'}")
    print(f"Worker Processes:   {args.num_workers}")
    print(f"Overwrite Existing: {args.overwrite}")
    print("-" * 80)

    t0 = time.perf_counter()
    real_meta = preprocess_domain(
        real_dir, real_out, res, args.prefix_frames, args.overwrite, args.num_workers, desc="Real Trajectories"
    )
    sim_meta = preprocess_domain(
        sim_dir, sim_out, res, args.prefix_frames, args.overwrite, args.num_workers, desc="Sim Trajectories"
    )
    elapsed = time.perf_counter() - t0

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resolution": list(res),
        "prefix_frames": args.prefix_frames,
        "real_count": len(real_meta),
        "sim_count": len(sim_meta),
        "real_dir": str(real_dir),
        "sim_dir": str(sim_dir),
        "real_trajectories": real_meta,
        "sim_trajectories": sim_meta,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_bytes = sum(m.get("bytes", 0) for m in list(real_meta.values()) + list(sim_meta.values()))
    print("=" * 80)
    print(f"Preprocessing completed in {elapsed:.2f} seconds!")
    print(f"Real trajectories converted: {len(real_meta)}")
    print(f"Sim trajectories converted:  {len(sim_meta)}")
    print(f"Total size on disk:          {total_bytes / 1e9:.2f} GB")
    print(f"Manifest written to:         {out_root / 'manifest.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

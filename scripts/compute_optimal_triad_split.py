#!/usr/bin/env python
"""Diagnostic script to compute optimal Macro/Micro rank partitions for Triad-AFNO.

Computes cumulative kinetic energy ratios and knee/elbow points from SVD singular
values of balanced normalized snapshots.
"""

import argparse
from pathlib import Path
import numpy as np
import torch

from pod_sim2real.data.arrow_dataset import ArrowWindowDataset, discover_trajectories
from pod_sim2real.data.splits import split_settings
from pod_sim2real.data.normalizer import GaussianNormalizer


def compute_modal_energy_stats(real_dir="data/data_real", sim_dir="data/data_sim", total_rank=64, resolution=(64, 128)):
    real_trajs = discover_trajectories(real_dir, "real")
    sim_trajs = discover_trajectories(sim_dir, "sim")
    r_tr, _ = split_settings(real_trajs, 1, 42)
    s_tr, _ = split_settings(sim_trajs, 1, 42)

    ds_r = ArrowWindowDataset(r_tr, 20, 20, 200, resolution)
    ds_s = ArrowWindowDataset(s_tr, 20, 20, 200, resolution)

    normalizer = GaussianNormalizer.from_dataset(ds_s, max_samples=32)

    snaps_u, snaps_v = [], []
    for ds in (ds_s, ds_r):
        count = min(len(ds), 32)
        for i in np.linspace(0, len(ds) - 1, count, dtype=int):
            x, y, _ = ds[int(i)]
            x, y = normalizer.preprocess(x, y)
            f = torch.cat([x, y], dim=0).numpy()
            snaps_u.append(f[:, 0].reshape(f.shape[0], -1))
            snaps_v.append(f[:, 1].reshape(f.shape[0], -1))

    Xu = np.concatenate(snaps_u, 0)
    Xv = np.concatenate(snaps_v, 0)

    _, su, _ = np.linalg.svd(Xu - Xu.mean(0), full_matrices=False)
    _, sv, _ = np.linalg.svd(Xv - Xv.mean(0), full_matrices=False)

    su_trunc = su[:total_rank]
    sv_trunc = sv[:total_rank]

    eu = np.cumsum(su_trunc**2) / np.sum(su_trunc**2)
    ev = np.cumsum(sv_trunc**2) / np.sum(sv_trunc**2)
    e_tot = np.cumsum(su_trunc**2 + sv_trunc**2) / np.sum(su_trunc**2 + sv_trunc**2)

    # Knee point calculation on log singular values
    log_s = np.log(su_trunc**2 + sv_trunc**2 + 1e-12)
    # Second difference (curvature)
    d2 = log_s[:-2] - 2 * log_s[1:-1] + log_s[2:]
    knee_rank = int(np.argmax(d2)) + 2

    return {
        "su": su_trunc,
        "sv": sv_trunc,
        "eu": eu,
        "ev": ev,
        "e_tot": e_tot,
        "knee_rank": knee_rank,
        "total_rank": total_rank,
    }


def find_split_by_threshold(e_tot, threshold=0.95):
    macro = int(np.searchsorted(e_tot, threshold)) + 1
    macro = max(4, min(len(e_tot) - 4, macro))
    micro = len(e_tot) - macro
    return macro, micro


def main():
    parser = argparse.ArgumentParser(description="Find optimal Macro/Micro rank partition for Triad-AFNO")
    parser.add_argument("--rank", type=int, default=64, help="Total rank (e.g. 64 or 96)")
    parser.add_argument("--real-dir", type=str, default="data/data_real")
    parser.add_argument("--sim-dir", type=str, default="data/data_sim")
    args = parser.parse_args()

    print(f"\n{'='*75}")
    print(f"Analyzing SVD Energy Distribution for Total Rank = {args.rank}")
    print(f"{'='*75}")

    stats = compute_modal_energy_stats(args.real_dir, args.sim_dir, total_rank=args.rank)
    e_tot, eu, ev = stats["e_tot"], stats["eu"], stats["ev"]

    header = f"{'Rank':<6} | {'Combined Energy':<16} | {'u Energy':<12} | {'v Energy':<12}"
    print(header)
    print("-" * len(header))
    checkpoints = [8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64]
    if args.rank > 64:
        checkpoints += [72, 80, 88, 96]
    checkpoints = [c for c in checkpoints if c <= args.rank]

    for k in checkpoints:
        print(f"{k:<6d} | {e_tot[k-1]*100:<15.2f}% | {eu[k-1]*100:<11.2f}% | {ev[k-1]*100:<11.2f}%")

    print(f"\n{'='*75}")
    print("Partition Recommendations (x = Macro, y = Micro):")
    print(f"{'='*75}")

    for thresh in [0.90, 0.95, 0.97, 0.98, 0.99]:
        m, mi = find_split_by_threshold(e_tot, thresh)
        print(f"  * Target {thresh*100:4.1f}% Energy: Macro = {m:2d} ({e_tot[m-1]*100:.2f}%), Micro = {mi:2d} (x + y = {args.rank})")

    print(f"  * Mathematical Knee Point: Macro = {stats['knee_rank']:2d}, Micro = {args.rank - stats['knee_rank']:2d}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()

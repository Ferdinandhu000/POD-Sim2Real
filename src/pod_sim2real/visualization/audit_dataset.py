from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..data import discover_trajectories
from ..data.pairing import pair_trajectories


def inspect(trajectory):
    row = {"domain": trajectory.domain, "sim_id": trajectory.sim_id, "re": trajectory.setting[0], "aoa": trajectory.setting[1], "source": trajectory.source, "shape": list(trajectory.shape), "channels": ["u", "v"] + (["p"] if trajectory.p is not None else []), "time_length": len(trajectory.t), "t_start": float(trajectory.t[0]), "t_end": float(trajectory.t[-1]), "dt_min": float(np.diff(trajectory.t).min()), "dt_max": float(np.diff(trajectory.t).max()), "coordinate_warning": False, "stats": {}}
    for name, array in [("u", trajectory.u), ("v", trajectory.v)] + ([ ("p", trajectory.p) ] if trajectory.p is not None else []):
        row["stats"][name] = {"nan": int(np.isnan(array).sum()), "inf": int(np.isinf(array).sum()), "min": float(np.nanmin(array)), "max": float(np.nanmax(array)), "mean": float(np.nanmean(array)), "std": float(np.nanstd(array))}
    for name, array in (("x", trajectory.x), ("y", trajectory.y)):
        if array is not None:
            row[name] = {"shape": list(array.shape), "min": float(np.nanmin(array)), "max": float(np.nanmax(array))}
            row["coordinate_warning"] |= not (-1e-3 <= np.nanmin(array) <= np.nanmax(array) <= 1.0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/audit"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    real = discover_trajectories(args.data_root / "data/data_real", "real")
    sim = discover_trajectories(args.data_root / "data/data_sim", "sim")
    pairs = pair_trajectories(real, sim)
    def source_parts(item):
        if item is None:
            return None, None
        source = str(item.source)
        path, sep, record = source.partition("#record=")
        return path, (int(record) if sep else 0)

    pair_rows = []
    for pair in pairs:
        real_arrow, real_record = source_parts(pair.real)
        sim_arrow, sim_record = source_parts(pair.sim)
        pair_rows.append({
            "re": pair.key[0], "aoa": pair.key[1],
            # These IDs are metadata stored inside Arrow; they often retain the
            # original RealPDEBench .h5 basename even though no .h5 is mounted.
            "real_id": pair.real.sim_id if pair.real else None,
            "sim_id": pair.sim.sim_id if pair.sim else None,
            "real_arrow": real_arrow, "real_record": real_record,
            "sim_arrow": sim_arrow, "sim_record": sim_record,
            "exact_id": pair.exact_id,
            "status": "matched" if pair.real and pair.sim else "unmatched",
        })
    report = {"data_root": str(args.data_root.resolve()), "real_count": len(real), "sim_count": len(sim), "matched_count": sum(row["status"] == "matched" for row in pair_rows), "trajectories": [inspect(item) for item in real + sim], "pairs": pair_rows, "notes": ["Matching is by normalized (Re,AoA) setting; equal sim_id does not imply identical physical trajectory.", "real_id/sim_id are logical identifiers embedded in Arrow records and may end in .h5; real_arrow/sim_arrow identify the actual Arrow files."]}
    (args.output_dir / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "pairing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]) if pair_rows else ["re", "aoa"])
        writer.writeheader(); writer.writerows(pair_rows)
    print(json.dumps({"report": str(args.output_dir / "dataset_report.json"), "real": len(real), "sim": len(sim), "matched": report["matched_count"]}, indent=2))


if __name__ == "__main__":
    main()

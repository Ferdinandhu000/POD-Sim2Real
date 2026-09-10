from __future__ import annotations

import argparse
import datetime
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ..data import OfficialArrowWindowDataset
from ..model import PODBasis, build_model
from .train import MODELS, _official_dataset, _resolve_data_layout
from .trainer import evaluate_model


def evaluate_all_checkpoints(
    checkpoints_dir: Path,
    data_root: Path | None = None,
    output_dir: Path | None = None,
    output_excel: str | None = None,
    device_name: str | None = None,
    batch_size: int = 16,
    num_workers: int = 0,
    prefix_frames_override: int | None = None,
    resolution_override: tuple[int, int] | list[int] | None = None,
) -> Path:
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"checkpoints directory not found: {checkpoints_dir}")

    # Discover candidate model directories
    # Look for folders containing best.pt directly or within subdirectories
    subdirs = sorted([p for p in checkpoints_dir.iterdir() if p.is_dir() and (p / "best.pt").exists()])
    if not subdirs:
        # Check if checkpoints_dir itself has best.pt
        if (checkpoints_dir / "best.pt").exists():
            subdirs = [checkpoints_dir]
        else:
            raise FileNotFoundError(f"no model folders with best.pt found in {checkpoints_dir}")

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Evaluating {len(subdirs)} model checkpoint(s) on device: {device}")

    summary_rows = []
    detailed_rows = []
    channel_rows = []
    full_results = {}

    for model_dir in subdirs:
        run_name = model_dir.name
        print(f"\n{'='*70}\nEvaluating checkpoint: {run_name}\n{'='*70}")

        cfg_path = model_dir / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})

        # Model name resolution
        model_name = model_cfg.get("name") or cfg.get("model_name")
        if not model_name:
            m = re.match(r"^(?:\d+_)?(.*?)(?:_config)?$", run_name)
            inferred = m.group(1).lower().replace("_", "-") if m else run_name
            if inferred in MODELS or inferred.replace("-", "_") in [x.replace("-", "_") for x in MODELS]:
                model_name = inferred
            elif "triad" in inferred:
                model_name = "triad-mno"
            elif "transolver" in inferred:
                if "pod" in inferred:
                    model_name = "pod-itransolver" if "itransolver" in inferred else "pod-transolver"
                else:
                    model_name = "itransolver"
            elif "itransformer" in inferred:
                model_name = "pod-itransformer"
            else:
                raise ValueError(f"cannot determine model name for {run_name}")

        input_steps = int(data_cfg.get("input_steps", 20))
        output_steps = int(data_cfg.get("output_steps", 20))
        raw_prefix = prefix_frames_override
        if raw_prefix is None:
            raw_prefix = data_cfg.get("prefix_frames", None)
        if raw_prefix is None:
            raw_prefix = data_cfg.get("prefix_ratio", None)
        if raw_prefix is None:
            raw_prefix = data_cfg.get("prefix_fraction", None)
        if raw_prefix is not None:
            val = float(raw_prefix)
            prefix_frames = val if val > 0 else None
        else:
            prefix_frames = None
        resolution = tuple(resolution_override or data_cfg.get("resolution", [64, 128]))

        # Resolve dataset directories
        root = Path(data_root or cfg.get("data_root") or Path.cwd())
        real_dir, sim_dir, index_root = _resolve_data_layout(root, data_cfg)

        # Load POD bases if present
        bases = None
        pod_u_path = model_dir / "pod_u.npz"
        pod_v_path = model_dir / "pod_v.npz"
        if not pod_u_path.exists():
            # Check pod_u_real.npz
            pod_u_path = model_dir / "pod_u_real.npz"
            pod_v_path = model_dir / "pod_v_real.npz"
        if pod_u_path.exists() and pod_v_path.exists():
            bases = (PODBasis.load(pod_u_path), PODBasis.load(pod_v_path))
            print(f"Loaded POD bases from {model_dir}")

        # Build model
        model_options = dict(model_cfg)
        model_options.pop("width", None)
        model_options.pop("name", None)
        model = build_model(
            model_name,
            bases,
            width=int(model_cfg.get("width", 32)),
            input_steps=input_steps,
            output_steps=output_steps,
            **model_options,
        )

        # Load weights
        ckpt_path = model_dir / "best.pt"
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = payload.get("model_state", payload.get("raw_model_state", payload))
        model.load_state_dict(state, strict=False)
        model.to(device).eval()

        # Evaluate across subsets
        subsets = ["all", "in_dist", "out_dist", "seen"] if index_root else ["all"]
        model_metrics = {}
        for sub in subsets:
            try:
                if index_root:
                    sub_ds = _official_dataset(
                        real_dir, index_root, "real", "test", input_steps, output_steps, resolution, test_mode=sub, prefix_frames=prefix_frames
                    )
                else:
                    from ..data.arrow_dataset import ArrowWindowDataset, discover_trajectories
                    real_trajs = discover_trajectories(real_dir, "real")
                    stride = int(data_cfg.get("stride", 20))
                    sub_ds = ArrowWindowDataset(real_trajs, input_steps, output_steps, stride, resolution)
                m = evaluate_model(model, sub_ds, device, batch_size, num_workers)
                m["windows"] = len(sub_ds)
                model_metrics[sub] = m

                detailed_rows.append({
                    "Model": run_name,
                    "Architecture": model_name,
                    "Subset": sub,
                    "Rel_L2": m["rel_l2"],
                    "MSE": m["mse"],
                    "RMSE": m["rmse"],
                    "MAE": m["mae"],
                    "Vorticity_MSE": m.get("vorticity_mse", float("nan")),
                    "Vorticity_Rel_L2": m.get("vorticity_rel_l2", float("nan")),
                    "Windows": m["windows"],
                })

                channel_rows.append({
                    "Model": run_name,
                    "Architecture": model_name,
                    "Subset": sub,
                    "U_Rel_L2": m.get("u_rel_l2", float("nan")),
                    "U_MSE": m.get("u_mse", float("nan")),
                    "U_RMSE": m.get("u_rmse", float("nan")),
                    "U_MAE": m.get("u_mae", float("nan")),
                    "V_Rel_L2": m.get("v_rel_l2", float("nan")),
                    "V_MSE": m.get("v_mse", float("nan")),
                    "V_RMSE": m.get("v_rmse", float("nan")),
                    "V_MAE": m.get("v_mae", float("nan")),
                })
            except Exception as e:
                print(f"Warning: could not evaluate subset {sub} for {run_name}: {e}")

        full_results[run_name] = model_metrics

        # Summary row (Table 1 structure)
        all_m = model_metrics.get("all", {})
        seen_m = model_metrics.get("seen", {})
        in_m = model_metrics.get("in_dist", {})
        out_m = model_metrics.get("out_dist", {})

        summary_rows.append({
            "Model": run_name,
            "Architecture": model_name,
            "Seen_Rel_L2": seen_m.get("rel_l2", float("nan")),
            "In_Dist_Rel_L2": in_m.get("rel_l2", float("nan")),
            "Out_Dist_Rel_L2": out_m.get("rel_l2", float("nan")),
            "Overall_Rel_L2": all_m.get("rel_l2", float("nan")),
            "RMSE": all_m.get("rmse", float("nan")),
            "MAE": all_m.get("mae", float("nan")),
            "Rel_L2_Vorticity": all_m.get("vorticity_rel_l2", float("nan")),
        })

    summary_df = pd.DataFrame(summary_rows)
    detailed_df = pd.DataFrame(detailed_rows)
    channel_df = pd.DataFrame(channel_rows)

    # Save to timestamped Excel file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir or Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = output_excel or f"evaluation_results_{timestamp}.xlsx"
    excel_path = out_dir / filename

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        detailed_df.to_excel(writer, sheet_name="Detailed_Subsets", index=False)
        channel_df.to_excel(writer, sheet_name="Per_Channel", index=False)

    json_path = excel_path.with_suffix(".json")
    json_path.write_text(json.dumps(full_results, indent=2), encoding="utf-8")

    print(f"\n{'='*95}")
    print(f"EVALUATION COMPLETE! Results saved to:")
    print(f"  Excel: {excel_path.resolve()}")
    print(f"  JSON:  {json_path.resolve()}")
    print(f"{'='*95}\n")
    print("Summary Table (Paper Table 1):")
    print(summary_df.to_string(index=False))
    print(f"\n{'='*95}")

    return excel_path


def main():
    parser = argparse.ArgumentParser(description="One-Click Evaluation of Best Checkpoints")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("best_checkpoints"), help="Path to best_checkpoints directory")
    parser.add_argument("--data-root", type=Path, help="Root directory containing dataset")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory to save Excel results")
    parser.add_argument("--output-excel", type=str, help="Custom filename for Excel output (default: evaluation_results_YYYYMMDD_HHMMSS.xlsx)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefix-frames", type=float, default=None, help="Frames or fraction to evaluate from start of trajectories (default: None for full sequence)")
    parser.add_argument("--resolution", nargs=2, type=int, default=None, help="Resolution override (H W)")
    args = parser.parse_args()

    evaluate_all_checkpoints(
        checkpoints_dir=args.checkpoints_dir,
        data_root=args.data_root,
        output_dir=args.output_dir,
        output_excel=args.output_excel,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefix_frames_override=args.prefix_frames,
        resolution_override=args.resolution,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..data import discover_trajectories
from ..data.pairing import pair_trajectories


def _panel(array: np.ndarray, lo: float, hi: float, size: tuple[int, int]) -> np.ndarray:
    image = np.clip((array - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    image = (255.0 * image).astype(np.uint8)
    image = Image.fromarray(image, mode="L").resize(size, Image.Resampling.NEAREST)
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[:, 0] = np.linspace(68, 253, 256)
    palette[:, 1] = np.linspace(1, 231, 256)
    palette[:, 2] = np.linspace(84, 37, 256)
    return palette[np.asarray(image)]


def _uv_arrays(trajectory, resolution: tuple[int, int] | None):
    if resolution is None:
        return trajectory.u, trajectory.v
    fields = trajectory.fields(resolution)
    return fields[:, 0], fields[:, 1]


def _speed_limits(datasets, frames: int, chunk_size: int = 100) -> tuple[float, float]:
    low, high = float("inf"), float("-inf")
    for u, v in datasets:
        for start in range(0, frames, chunk_size):
            stop = min(start + chunk_size, frames)
            speed = np.sqrt(u[start:stop] ** 2 + v[start:stop] ** 2)
            low = min(low, float(speed.min()))
            high = max(high, float(speed.max()))
    return low, high


def _domain_limits(uv, frames: int) -> list[tuple[float, float]]:
    limits = [
        (float(uv[channel][:frames].min()), float(uv[channel][:frames].max()))
        for channel in range(2)
    ]
    limits.append(_speed_limits((uv,), frames))
    return limits


def render_pair(pair, output_dir: Path, fps: float = 12.5, max_frames: int = 0, resolution: tuple[int, int] | None = None) -> dict:
    real, sim = pair.real, pair.sim
    output_dir.mkdir(parents=True, exist_ok=True)
    real_uv, sim_uv = _uv_arrays(real, resolution), _uv_arrays(sim, resolution)
    n = min(len(real_uv[0]), len(sim_uv[0]))
    if max_frames:
        n = min(n, max_frames)
    # Use stable full-trajectory limits per domain. Real and Sim can have very
    # different magnitudes, so sharing limits can hide the lower-variance row.
    domain_limits = {"Real": _domain_limits(real_uv, n), "Sim": _domain_limits(sim_uv, n)}
    # The source grid is 2:1 (256 x 128). A 2x display preserves that ratio
    # without discarding any source cells.
    panel_w, panel_h = 512, 256
    path = output_dir / f"pair_Re{pair.key[0]}_AoA{pair.key[1]:g}.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render MP4 videos")
    process = subprocess.Popen([ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{panel_w * 3}x{panel_h * 2}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for index in range(n):
        rows = []
        for uv, trajectory, domain in ((real_uv, real, "Real"), (sim_uv, sim, "Sim")):
            values = (uv[0][index], uv[1][index], np.sqrt(uv[0][index] ** 2 + uv[1][index] ** 2))
            panels = []
            for channel_index, (channel, value) in enumerate(zip(("u", "v", "speed"), values)):
                lo, hi = domain_limits[domain][channel_index]
                image = Image.fromarray(_panel(value, lo, hi, (panel_w, panel_h)), mode="RGB")
                label = f"{domain} {channel} | frame {index:04d} | t={trajectory.t[index]:.5f} | [{lo:.3g}, {hi:.3g}]"
                ImageDraw.Draw(image).text((8, 8), label, fill="white")
                panels.append(np.asarray(image))
            rows.append(np.hstack(panels))
        process.stdin.write(np.vstack(rows).tobytes())
    process.stdin.close()
    error = process.stderr.read().decode("utf-8", errors="replace")
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {error[-1000:]}")
    source_resolution = [int(real.u.shape[1]), int(real.u.shape[2])]
    serializable_limits = {domain.lower(): [[lo, hi] for lo, hi in limits] for domain, limits in domain_limits.items()}
    return {"video": str(path), "frames": n, "fps": fps, "source_resolution": source_resolution, "render_resolution": list(resolution) if resolution else source_resolution, "color_limits": serializable_limits, "color_scale": "fixed_per_domain_per_channel", "re": pair.key[0], "aoa": pair.key[1], "real_id": real.sim_id, "sim_id": sim.sim_id, "real_arrow": real.source, "sim_arrow": sim.source}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/videos"))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=12.5)
    parser.add_argument("--resolution", nargs=2, type=int, metavar=("H", "W"), help="optional spatial downsampling; omitted means original resolution")
    args = parser.parse_args()
    real = discover_trajectories(args.data_root / "data/data_real", "real")
    sim = discover_trajectories(args.data_root / "data/data_sim", "sim")
    resolution = tuple(args.resolution) if args.resolution else None
    entries = [render_pair(pair, args.output_dir, args.fps, args.max_frames, resolution) for pair in pair_trajectories(real, sim) if pair.real and pair.sim]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "video_manifest.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(json.dumps({"videos": len(entries), "manifest": str(args.output_dir / "video_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()

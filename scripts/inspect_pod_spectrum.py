import argparse
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from pod_sim2real.data.arrow_dataset import ArrowWindowDataset, discover_trajectories
from pod_sim2real.data.splits import split_settings


def load_balanced_snapshots(real_dir, sim_dir, resolution=(64, 128), max_samples=128, stride=200):
    real_trajs = discover_trajectories(real_dir, "real")
    sim_trajs = discover_trajectories(sim_dir, "sim")
    real_train, _ = split_settings(real_trajs, 1, 42)
    sim_train, _ = split_settings(sim_trajs, 1, 42)

    real_ds = ArrowWindowDataset(real_train, 20, 20, stride, resolution)
    sim_ds = ArrowWindowDataset(sim_train, 20, 20, stride, resolution)

    u_samples = []
    v_samples = []

    for name, ds in [("real", real_ds), ("sim", sim_ds)]:
        n_take = min(len(ds), max_samples // 2)
        indices = np.linspace(0, len(ds) - 1, n_take, dtype=int)
        for idx in indices:
            x, y, _ = ds[int(idx)]
            fields = torch.cat((x, y), dim=0).numpy()  # [40, 2, H, W]
            u_flat = fields[:, 0].reshape(fields.shape[0], -1)  # [40, H*W]
            v_flat = fields[:, 1].reshape(fields.shape[0], -1)  # [40, H*W]
            u_samples.append(u_flat)
            v_samples.append(v_flat)

    X_u = np.concatenate(u_samples, axis=0).astype(np.float32)  # [N_snaps, H*W]
    X_v = np.concatenate(v_samples, axis=0).astype(np.float32)
    return X_u, X_v


def analyze_spectrum(X, channel_name="u", max_rank=128):
    mean = X.mean(axis=0)
    X_cent = X - mean
    U, s, Vt = np.linalg.svd(X_cent, full_matrices=False)
    lambdas = s ** 2
    total_energy = np.sum(lambdas)
    cum_energy = np.cumsum(lambdas) / total_energy

    # Projection reconstruction error on snapshots: ||x - P_k x|| / ||x||
    recon_errs = []
    ranks = [8, 16, 24, 32, 48, 64, 80, 96, 112, 128]
    ranks = [r for r in ranks if r <= len(s)]

    denom = np.linalg.norm(X)
    for r in ranks:
        P_r = Vt[:r]  # [r, H*W]
        coeff = X_cent @ P_r.T  # [N, r]
        recon = coeff @ P_r + mean  # [N, H*W]
        rel_l2 = np.linalg.norm(X - recon) / denom
        recon_errs.append(rel_l2)

    return {
        "channel": channel_name,
        "s": s,
        "lambdas": lambdas,
        "cum_energy": cum_energy,
        "ranks": ranks,
        "recon_errs": recon_errs,
        "mean": mean,
        "modes": Vt[:max_rank],
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect POD energy spectrum and modal cutoff")
    parser.add_argument("--real-dir", default="data/data_real", type=str)
    parser.add_argument("--sim-dir", default="data/data_sim", type=str)
    parser.add_argument("--resolution", default="64,128", type=str)
    parser.add_argument("--out-dir", default="analysis_v3", type=str)
    args = parser.parse_args()

    h, w = map(int, args.resolution.split(","))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print("Loading balanced snapshots from Sim + Real training sets...")
    X_u, X_v = load_balanced_snapshots(args.real_dir, args.sim_dir, resolution=(h, w), max_samples=64)
    print(f"Loaded snapshots: u shape = {X_u.shape}, v shape = {X_v.shape}")

    res_u = analyze_spectrum(X_u, "u (Streamwise)")
    res_v = analyze_spectrum(X_v, "v (Transverse)")

    # -------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["figure.dpi"] = 300
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Singular value decay
    ax = axes[0]
    ranks_plot = np.arange(1, min(129, len(res_u["s"]) + 1))
    ax.semilogy(ranks_plot, res_u["s"][:len(ranks_plot)], label="u velocity", color="#1f77b4", lw=2)
    ax.semilogy(ranks_plot, res_v["s"][:len(ranks_plot)], label="v velocity", color="#ff7f0e", lw=2)
    ax.set_title("Singular Value Decay (Log Scale)", fontsize=12, fontweight="bold")
    ax.set_xlabel("POD Mode Index $k$", fontsize=11)
    ax.set_ylabel("Singular Value $\sigma_k$", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)

    # Plot 2: Cumulative Energy Ratio
    ax = axes[1]
    ax.plot(ranks_plot, res_u["cum_energy"][:len(ranks_plot)] * 100, label="u cumulative energy", color="#1f77b4", lw=2)
    ax.plot(ranks_plot, res_v["cum_energy"][:len(ranks_plot)] * 100, label="v cumulative energy", color="#ff7f0e", lw=2)
    ax.axhline(90, color="gray", linestyle="--", alpha=0.7, label="90% threshold")
    ax.axhline(95, color="purple", linestyle="--", alpha=0.7, label="95% threshold")
    ax.axhline(99, color="darkred", linestyle="--", alpha=0.7, label="99% threshold")
    ax.axvline(32, color="blue", linestyle=":", alpha=0.7, label="Rank 32")
    ax.axvline(64, color="green", linestyle=":", alpha=0.7, label="Rank 64")
    ax.set_title("Cumulative Energy Ratio (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Modes $K$", fontsize=11)
    ax.set_ylabel("Cumulative Variance / Energy (%)", fontsize=11)
    ax.set_ylim(50, 100.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9, loc="lower right")

    # Plot 3: Orthogonal Projection Reconstruction Error
    ax = axes[2]
    ax.plot(res_u["ranks"], [e * 100 for e in res_u["recon_errs"]], marker="o", label="u Recon Rel-L2", color="#1f77b4", lw=2)
    ax.plot(res_v["ranks"], [e * 100 for e in res_v["recon_errs"]], marker="s", label="v Recon Rel-L2", color="#ff7f0e", lw=2)
    ax.set_title("Theoretical POD Truncation Error Floor", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Modes $K$", fontsize=11)
    ax.set_ylabel("Relative Reconstruction L2 Error (%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plot_path = out_dir / "pod_energy_spectrum_analysis.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved analysis plot to {plot_path}")

    # Print summary table
    print("\n=========================================================================================")
    print("                      POD ENERGY & RECONSTRUCTION SPECTRUM TABLE                         ")
    print("=========================================================================================")
    print(f"{'Rank K':<8} | {'u Cum Energy (%)':<18} | {'v Cum Energy (%)':<18} | {'u Recon Error (%)':<18} | {'v Recon Error (%)':<18}")
    print("-----------------------------------------------------------------------------------------")
    for r, eu, ev in zip(res_u["ranks"], res_u["recon_errs"], res_v["recon_errs"]):
        idx = min(r - 1, len(res_u["cum_energy"]) - 1)
        ce_u = res_u["cum_energy"][idx] * 100
        ce_v = res_v["cum_energy"][idx] * 100
        print(f"{r:<8} | {ce_u:<18.2f} | {ce_v:<18.2f} | {eu*100:<18.2f} | {ev*100:<18.2f}")
    print("=========================================================================================\n")


if __name__ == "__main__":
    main()

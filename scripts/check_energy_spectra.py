#!/usr/bin/env python
"""Session 2 gate check: energy spectra across viscosity levels.

Loads data/visc_sweep_v2/velocity_nu{i}.nc, computes spherically-averaged
2D kinetic energy spectrum E(k) for each ν level, and saves a log-log plot.

Visual gate (human review):
  - High-ν (laminar): energy concentrated near k_forcing=4, steep drop-off.
  - Low-ν  (turbulent): broader spectrum, shallower slope (enstrophy cascade ~k^-3,
    possibly inverse-cascade range ~k^-5/3 for k < k_f).

Usage:
    uv run python scripts/check_energy_spectra.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

DATA_DIR  = Path(__file__).resolve().parent.parent / "data" / "visc_sweep_v2"
FIG_DIR   = Path(__file__).resolve().parent.parent / "figures"
FIG_PATH  = FIG_DIR / "energy_spectra_visc_sweep.png"

N_SAMPLE_SNAPS = 10   # snapshots to average per ν level (last trajectory)
K_FORCE = 4


# ── Spherically-averaged 2D energy spectrum ───────────────────────────────────
def energy_spectrum(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """E(k) from one (H, W) velocity snapshot.

    Returns (k_bins, E_k) where k_bins are integer wavenumber shells
    and E_k is kinetic energy per shell.
    """
    N = u.shape[0]
    u_hat = np.fft.fft2(u.astype(np.float64))
    v_hat = np.fft.fft2(v.astype(np.float64))
    E_2d = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2) / N ** 4

    kx_1d = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kx_1d, kx_1d, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2)
    k_int = np.round(K).astype(int)

    k_max = N // 2
    k_bins = np.arange(1, k_max + 1)
    E_k = np.array([E_2d[k_int == k].sum() for k in k_bins])
    return k_bins, E_k


def load_nu_values(data_dir: Path) -> list[float]:
    nu_vals = []
    for i in range(5):
        p = data_dir / f"velocity_nu{i}.nc"
        ds = xr.open_dataset(p)
        nu_vals.append(float(ds.attrs["nu"]))
        ds.close()
    return nu_vals


def compute_mean_spectrum(nc_path: Path, n_snaps: int) -> tuple[np.ndarray, np.ndarray]:
    """Average E(k) over the last n_snaps snapshots of the last trajectory."""
    ds = xr.open_dataset(nc_path)
    vel = ds["velocity"]   # (sample, time, channel, x, y)
    n_traj = vel.sizes["sample"]
    n_time = vel.sizes["time"]

    # Use last trajectory; last n_snaps timesteps
    traj = vel[n_traj - 1, max(0, n_time - n_snaps):, :, :, :].values  # (T, C, H, W)
    ds.close()

    E_all = []
    for t in range(traj.shape[0]):
        u = traj[t, 0]
        v = traj[t, 1]
        k_bins, E_k = energy_spectrum(u, v)
        E_all.append(E_k)

    return k_bins, np.mean(E_all, axis=0)


# ── Plotting ──────────────────────────────────────────────────────────────────
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Check all files present
    missing = [f"velocity_nu{i}.nc" for i in range(5)
               if not (DATA_DIR / f"velocity_nu{i}.nc").exists()]
    if missing:
        print(f"Missing files: {missing}")
        print("Run gen_visc_sweep.py first.")
        return

    nu_values = load_nu_values(DATA_DIR)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, 5))

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, nu in enumerate(nu_values):
        nc_path = DATA_DIR / f"velocity_nu{i}.nc"
        k_bins, E_k = compute_mean_spectrum(nc_path, N_SAMPLE_SNAPS)

        label = f"ν={nu:.2e}"
        if i == 2:
            label += " (holdout)"
        ax.loglog(k_bins, E_k, color=colors[i], lw=1.8, label=label)

    # Reference slopes
    k_ref = np.array([2, 30], dtype=float)
    # Enstrophy cascade k^-3 (2D turbulence, k > k_f)
    E0_3 = E_k[K_FORCE]   # anchor at k_f region (rough)
    ax.loglog(k_ref, E0_3 * (k_ref / K_FORCE) ** (-3),
              "k--", lw=1, alpha=0.6, label=r"$k^{-3}$ (enstrophy cascade)")
    # Inverse cascade k^-5/3 (k < k_f)
    k_inv = np.array([1, K_FORCE], dtype=float)
    ax.loglog(k_inv, E0_3 * (k_inv / K_FORCE) ** (-5 / 3),
              "k:", lw=1, alpha=0.6, label=r"$k^{-5/3}$ (inverse cascade)")

    ax.axvline(K_FORCE, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax.text(K_FORCE * 1.05, ax.get_ylim()[0] * 2, f"$k_f={K_FORCE}$",
            color="gray", fontsize=9)

    ax.set_xlabel("Wavenumber $k$")
    ax.set_ylabel("$E(k)$")
    ax.set_title("Energy spectra — viscosity sweep\n"
                 "(Gate: high-ν steep, low-ν shallow/Kolmogorov-like)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    print(f"Saved → {FIG_PATH}")

    # Print summary statistics
    print("\nSummary (mean E(k) integrated over k=1..N//2):")
    for i, nu in enumerate(nu_values):
        nc_path = DATA_DIR / f"velocity_nu{i}.nc"
        _, E_k = compute_mean_spectrum(nc_path, N_SAMPLE_SNAPS)
        print(f"  ν[{i}]={nu:.2e}  E_total={E_k.sum():.4e}  "
              f"E(k=1)={E_k[0]:.4e}  E(k={K_FORCE})={E_k[K_FORCE-1]:.4e}")

    print("\nGate check: open figures/energy_spectra_visc_sweep.png")
    print("  PASS if: high-ν curves decay steeply from k_f; "
          "low-ν curves extend to small k with shallower slope.")


if __name__ == "__main__":
    main()

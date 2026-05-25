#!/usr/bin/env python
"""Session 2: Generate auxiliary viscosity-sweep dataset.

Pseudo-spectral 2D incompressible NS, vorticity-streamfunction formulation.
Periodic BCs on [0, 2π]².  Kolmogorov forcing F_x = A·sin(k_f·y).
Ekman drag -α·ω arrests the 2D inverse energy cascade.

5 viscosity levels (log-uniform), 200 trajectories each.
Output: data/visc_sweep_v2/velocity_nu{i}.nc  shape (200, 21, 3, 128, 128).

Usage:
    uv run python scripts/gen_visc_sweep.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

np.random.seed(42)

# ── Grid ──────────────────────────────────────────────────────────────────────
N = 128
L = 2 * np.pi

# ── Viscosity levels (log-uniform; index 2 is the holdout per CLAUDE.md) ─────
NU_VALUES = np.logspace(-4, -2, 5)  # [1e-4, ~3.2e-4, 1e-3, ~3.2e-3, 1e-2]

# ── Forcing & drag ─────────────────────────────────────────────────────────────
K_FORCE = 4      # forcing wavenumber
A_FORCE = 1.0    # forcing amplitude
ALPHA   = 0.01   # Ekman drag coefficient (arrests inverse cascade)

# ── Time parameters ───────────────────────────────────────────────────────────
DT       = 0.002                          # RK4 timestep
T_SPINUP = 50.0                           # spin-up duration
N_TRAJS  = 200                            # trajectories per ν level
N_SNAPS  = 21                             # snapshots per trajectory
DT_SNAP  = 0.05                           # time between snapshots
T_GAP    = 1.0                            # gap between trajectory windows

SNAP_STEPS = int(round(DT_SNAP / DT))    # = 25
GAP_STEPS  = int(round(T_GAP  / DT))     # = 500
SPIN_STEPS = int(round(T_SPINUP / DT))   # = 25 000

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "visc_sweep_v2"


# ── Spectral setup ────────────────────────────────────────────────────────────
def setup_spectral(N: int):
    kx_1d = np.fft.rfftfreq(N, d=1.0 / N)   # (N//2+1,)  [0,1,…,N//2]
    ky_1d = np.fft.fftfreq(N, d=1.0 / N)    # (N,)       [0,1,…,-1]
    kx = kx_1d[np.newaxis, :]               # (1, N//2+1)
    ky = ky_1d[:, np.newaxis]               # (N, 1)
    k2 = kx ** 2 + ky ** 2                  # (N, N//2+1)
    k2_safe = k2.copy()
    k2_safe[0, 0] = 1.0                     # avoid /0; psi[0,0] forced to 0 later
    kmax = N // 3                            # 2/3 dealiasing
    dealias = ((np.abs(kx) <= kmax) & (np.abs(ky) <= kmax)).astype(np.float64)
    return kx, ky, k2_safe, dealias


def make_forcing_hat(N: int, k_force: int, amplitude: float) -> np.ndarray:
    """Spectral vorticity forcing from F_x = A·sin(k_f·y).
    curl_z(F_x, 0) = -∂F_x/∂y = -A·k_f·cos(k_f·y)
    """
    y = np.linspace(0, L, N, endpoint=False)
    force_y = -amplitude * k_force * np.cos(k_force * y)       # (N,)
    force_2d = np.tile(force_y[:, np.newaxis], (1, N))          # (N, N)
    return np.fft.rfft2(force_2d)


# ── Vorticity RHS ─────────────────────────────────────────────────────────────
def rhs_omega(omega_hat, kx, ky, k2, nu, forcing_hat, dealias):
    psi_hat = -omega_hat / k2
    psi_hat[0, 0] = 0.0

    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat
    dw_dx_hat = 1j * kx * omega_hat
    dw_dy_hat = 1j * ky * omega_hat

    u     = np.fft.irfft2(dealias * u_hat)
    v     = np.fft.irfft2(dealias * v_hat)
    dw_dx = np.fft.irfft2(dealias * dw_dx_hat)
    dw_dy = np.fft.irfft2(dealias * dw_dy_hat)

    nl = -(u * dw_dx + v * dw_dy)
    return (dealias * np.fft.rfft2(nl)
            - nu * k2 * omega_hat
            - ALPHA * omega_hat
            + forcing_hat)


def rk4_omega(omega_hat, kx, ky, k2, nu, forcing_hat, dealias, dt):
    k1 = rhs_omega(omega_hat,              kx, ky, k2, nu, forcing_hat, dealias)
    k2_ = rhs_omega(omega_hat + .5*dt*k1,  kx, ky, k2, nu, forcing_hat, dealias)
    k3 = rhs_omega(omega_hat + .5*dt*k2_,  kx, ky, k2, nu, forcing_hat, dealias)
    k4 = rhs_omega(omega_hat +    dt*k3,   kx, ky, k2, nu, forcing_hat, dealias)
    return dealias * (omega_hat + (dt / 6.0) * (k1 + 2*k2_ + 2*k3 + k4))


# ── Passive tracer (RK4; Forward Euler is unconditionally unstable for advection) ─
def rhs_tracer(theta_hat, psi_hat, kx, ky, k2, kappa, dealias):
    """Tracer RHS. psi_hat = -omega_hat/k2 pre-computed by caller."""
    u = np.fft.irfft2(dealias * (1j * ky * psi_hat))
    v = np.fft.irfft2(dealias * (-1j * kx * psi_hat))
    dth_dx = np.fft.irfft2(dealias * (1j * kx * theta_hat))
    dth_dy = np.fft.irfft2(dealias * (1j * ky * theta_hat))
    nl = -(u * dth_dx + v * dth_dy)
    return dealias * np.fft.rfft2(nl) - kappa * k2 * theta_hat


def rk4_tracer(theta_hat, omega_hat, kx, ky, k2, kappa, dealias, dt):
    """RK4 tracer step with omega frozen at start of step (first-order splitting)."""
    psi_hat = -omega_hat / k2
    psi_hat[0, 0] = 0.0
    k1  = rhs_tracer(theta_hat,              psi_hat, kx, ky, k2, kappa, dealias)
    k2_ = rhs_tracer(theta_hat + .5*dt*k1,  psi_hat, kx, ky, k2, kappa, dealias)
    k3  = rhs_tracer(theta_hat + .5*dt*k2_, psi_hat, kx, ky, k2, kappa, dealias)
    k4  = rhs_tracer(theta_hat + dt*k3,     psi_hat, kx, ky, k2, kappa, dealias)
    return dealias * (theta_hat + (dt / 6.0) * (k1 + 2*k2_ + 2*k3 + k4))


# ── Recover u, v from omega ────────────────────────────────────────────────────
def omega_to_uv(omega_hat, kx, ky, k2, dealias):
    psi_hat = -omega_hat / k2
    psi_hat[0, 0] = 0.0
    u = np.fft.irfft2(dealias * (1j * ky * psi_hat)).real.astype(np.float32)
    v = np.fft.irfft2(dealias * (-1j * kx * psi_hat)).real.astype(np.float32)
    return u, v


# ── Random initial conditions ──────────────────────────────────────────────────
def random_ic(N: int, rng, kx, ky, k2, dealias):
    """Low-wavenumber random vorticity (1/k energy spectrum)."""
    phase = rng.uniform(0, 2 * np.pi, (N, N // 2 + 1))
    amp   = rng.standard_normal((N, N // 2 + 1))
    weight = np.where(k2 > 1.0, 1.0 / np.sqrt(k2), 0.0)
    omega_hat = amp * weight * np.exp(1j * phase)
    omega_hat = dealias * omega_hat
    omega_hat[0, 0] = 0.0   # zero-mean vorticity
    return omega_hat


# ── NaN guard ─────────────────────────────────────────────────────────────────
def check_nan(arr, label: str):
    if np.any(~np.isfinite(arr)):
        raise RuntimeError(f"Non-finite values in {label}. "
                           "Possible instability — reduce DT or A_FORCE.")


# ── Main simulation per ν level ────────────────────────────────────────────────
def generate_for_nu(nu: float, nu_idx: int, rng, kx, ky, k2, dealias, forcing_hat):
    kappa = nu   # Schmidt number Sc = 1

    omega_hat = random_ic(N, rng, kx, ky, k2, dealias)
    theta_ic  = rng.standard_normal((N, N)) * 0.1
    theta_hat = dealias * np.fft.rfft2(theta_ic)

    # ── Spin-up ────────────────────────────────────────────────────────────────
    for si in tqdm(range(SPIN_STEPS), desc=f"  ν[{nu_idx}]={nu:.2e} spinup", leave=False):
        omega_hat = rk4_omega(omega_hat, kx, ky, k2, nu, forcing_hat, dealias, DT)
        theta_hat = rk4_tracer(theta_hat, omega_hat, kx, ky, k2, kappa, dealias, DT)
        if si % 2000 == 1999:
            check_nan(omega_hat, f"omega_hat at spinup step {si}, ν={nu:.2e}")

    check_nan(omega_hat, f"omega_hat after spinup ν={nu:.2e}")
    check_nan(theta_hat, f"theta_hat after spinup ν={nu:.2e}")

    psi_tmp = -omega_hat / k2; psi_tmp[0, 0] = 0.0
    u_rms = float(np.sqrt(np.mean(np.fft.irfft2(dealias * (1j * ky * psi_tmp)) ** 2)))
    print(f"    spinup done — U_rms ≈ {u_rms:.4f}")

    # ── Extract trajectory windows ─────────────────────────────────────────────
    data = np.zeros((N_TRAJS, N_SNAPS, 3, N, N), dtype=np.float32)

    for i_traj in tqdm(range(N_TRAJS), desc=f"  ν[{nu_idx}]={nu:.2e} trajs", leave=False):
        for i_snap in range(N_SNAPS):
            u, v = omega_to_uv(omega_hat, kx, ky, k2, dealias)
            theta = np.fft.irfft2(theta_hat).real.astype(np.float32)
            data[i_traj, i_snap, 0] = u
            data[i_traj, i_snap, 1] = v
            data[i_traj, i_snap, 2] = theta
            for ch, label in enumerate(("u", "v", "theta")):
                if np.any(~np.isfinite(data[i_traj, i_snap, ch])):
                    raise RuntimeError(
                        f"Non-finite {label}: traj={i_traj} snap={i_snap} ν={nu:.2e}"
                    )
            if i_snap < N_SNAPS - 1:
                for _ in range(SNAP_STEPS):
                    omega_hat = rk4_omega(omega_hat, kx, ky, k2, nu, forcing_hat, dealias, DT)
                    theta_hat = rk4_tracer(theta_hat, omega_hat, kx, ky, k2, kappa, dealias, DT)

        # Gap between trajectories
        for _ in range(GAP_STEPS):
            omega_hat = rk4_omega(omega_hat, kx, ky, k2, nu, forcing_hat, dealias, DT)
            theta_hat = rk4_tracer(theta_hat, omega_hat, kx, ky, k2, kappa, dealias, DT)

    return data


# ── Save in PDEgym format ──────────────────────────────────────────────────────
def save_nc(data: np.ndarray, nu: float, nu_idx: int, out_dir: Path):
    out_path = out_dir / f"velocity_nu{nu_idx}.nc"
    ds = xr.Dataset(
        {"velocity": (["sample", "time", "channel", "x", "y"], data)},
        attrs={
            "nu": float(nu),
            "nu_idx": nu_idx,
            "nu_values": list(NU_VALUES),
            "forcing": f"Kolmogorov F_x=A*sin(k_f*y), A={A_FORCE}, k_f={K_FORCE}",
            "ekman_drag": ALPHA,
            "domain": "periodic [0, 2pi]^2",
            "grid": f"{N}x{N}",
            "dt": DT,
            "snapshots": N_SNAPS,
            "dt_snap": DT_SNAP,
            "generator": "scripts/gen_visc_sweep.py",
        },
    )
    encoding = {"velocity": {"dtype": "float32", "zlib": True, "complevel": 1}}
    ds.to_netcdf(out_path, format="NETCDF4", encoding=encoding)
    size_gb = out_path.stat().st_size / 1e9
    print(f"    Saved → {out_path} ({size_gb:.2f} GB on disk)")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    kx, ky, k2, dealias = setup_spectral(N)
    forcing_hat = make_forcing_hat(N, K_FORCE, A_FORCE)
    rng = np.random.default_rng(42)

    print("Session 2 — viscosity sweep dataset generation")
    print(f"  ν values : {NU_VALUES}")
    print(f"  Holdout  : ν[2] = {NU_VALUES[2]:.4e}")
    print(f"  Grid     : {N}×{N},  L=2π,  dt={DT}")
    print(f"  Forcing  : A={A_FORCE}, k_f={K_FORCE}, α_Ekman={ALPHA}")
    print(f"  Spinup   : {T_SPINUP} time-units ({SPIN_STEPS} steps)")
    print(f"  Output   : {N_TRAJS} trajs × {N_SNAPS} snaps × {len(NU_VALUES)} ν levels")
    print()

    total_t0 = time.time()
    for nu_idx, nu in enumerate(NU_VALUES):
        out_path = OUT_DIR / f"velocity_nu{nu_idx}.nc"
        if out_path.exists():
            print(f"ν[{nu_idx}]={nu:.2e} — file exists, skipping.")
            continue
        print(f"ν[{nu_idx}]={nu:.2e} ─────────────────────────")
        t0 = time.time()
        data = generate_for_nu(nu, nu_idx, rng, kx, ky, k2, dealias, forcing_hat)
        save_nc(data, nu, nu_idx, OUT_DIR)
        print(f"    elapsed: {(time.time()-t0)/60:.1f} min")

    print(f"\nAll done.  Total time: {(time.time()-total_t0)/60:.1f} min")
    print("Next: run scripts/check_energy_spectra.py for gate check.")


if __name__ == "__main__":
    main()

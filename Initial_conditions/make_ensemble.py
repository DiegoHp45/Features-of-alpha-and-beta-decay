"""
make_ensemble.py
================

Ensemble of parametric NSM-ejecta trajectories for WinNet.

Free parameters per trajectory (sampled with a scrambled Sobol sequence,
2^m points, uniform in each range):
    Ye0   in [0.04, 0.42]
    s0    in [1, 49]        k_B / baryon
    v_inf in [0.01, 0.49] c

Derived:
    rho0  : s_tot(T0, rho0, Ye0) = s0 with WinNet-data NSE + Helmholtz EOS
            (see initial_state.py)
    tau   : late-time homologous branch R = R0 e t / (3 tau) moves at
            v_inf  =>  tau = e R0 / (3 v_inf)

Density (Lippuner & Roberts 2015, Eq. 1):
    rho = rho0 exp(-t/tau)               t <= 3 tau
    rho = rho0 (3 tau / (e t))^3          t >= 3 tau
Radius from mass conservation, R = R0 (rho0/rho)^(1/3):
    R = R0 exp(t / 3tau)                 t <= 3 tau
    R = R0 e t / (3 tau)                  t >= 3 tau
Temperature column: adiabatic placeholder T0 (rho/rho0)^(1/3); with
heating_mode = 1 WinNet recomputes T from the entropy.  Ye is constant
(WinNet evolves it with the weak rates).

Files are written to t_end = 1 s; beyond that WinNet's expansiontype = 1
(rho ~ t^-3) continues the late branch exactly.

Usage:  ../.venv/bin/python make_ensemble.py [--m 10] [--seed 42] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from initial_state import GK, InitialStateSolver

C_KM = 2.99792458e5               # km/s

T0 = 7.0 * GK
R0_KM = 50.0
RANGES = {"Ye0": (0.04, 0.42), "s0": (1.0, 49.0), "v_inf_c": (0.01, 0.49)}
RHO0_FLAG = 1.0e12               # above: neutron degeneracy / EOS limits


def tau_from_vinf(v_inf_c, R0_km=R0_KM):
    return np.e * R0_km / (3.0 * v_inf_c * C_KM)


def rho_of_t(t, rho0, tau):
    t = np.asarray(t, float)
    return np.where(t <= 3 * tau, rho0 * np.exp(-np.minimum(t, 3 * tau) / tau),
                    rho0 * (3 * tau / (np.e * np.maximum(t, 3 * tau))) ** 3)


def R_of_t(t, tau, R0_km=R0_KM):
    t = np.asarray(t, float)
    return np.where(t <= 3 * tau, R0_km * np.exp(np.minimum(t, 3 * tau) / (3 * tau)),
                    R0_km * np.e * t / (3 * tau))


def time_grid(tau, t_end=1.0, npts=400):
    """t = 0, then log-spaced from 1e-3 tau to t_end, with t = 3 tau exact."""
    t = np.concatenate([[0.0], np.geomspace(1e-3 * tau, t_end, npts - 2),
                        [3 * tau]])
    return np.unique(t)


def sample(m=10, seed=42):
    u = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(m)
    lo = np.array([r[0] for r in RANGES.values()])
    hi = np.array([r[1] for r in RANGES.values()])
    return qmc.scale(u, lo, hi)


def write_traj(path, t, T9, rho, R, Ye, note):
    M = np.column_stack([t, T9, rho, R, np.full_like(t, Ye)])
    np.savetxt(path, M, fmt="%18.10e",
               header=note + "\ntime[s]  T9[GK]  rho[g/cm3]  R[km]  Ye")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=10, help="2^m trajectories")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent
                                         / "Trajectories_ensemble"))
    ap.add_argument("--t_end", type=float, default=1.0)
    ap.add_argument("--npts", type=int, default=400)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P = sample(a.m, a.seed)
    solver = InitialStateSolver(T0)
    rows, t_start = [], time.time()

    for i, (Ye, s0, v) in enumerate(P, start=1):
        st = solver.rho0(s0, Ye)
        rho0, tau = st["rho0"], tau_from_vinf(v)
        t = time_grid(tau, a.t_end, a.npts)
        rho = rho_of_t(t, rho0, tau)
        R = R_of_t(t, tau)
        T9 = (T0 / GK) * (rho / rho0) ** (1.0 / 3.0)
        fname = f"traj{i:04d}.dat"
        write_traj(out / fname, t, T9, rho, R, Ye,
                   f"Ye0={Ye:.5f} s0={s0:.4f} kB/baryon v_inf={v:.4f}c "
                   f"tau={tau*1e3:.5f} ms rho0={rho0:.5e} g/cm3 "
                   f"T0={T0/GK:.1f} GK R0={R0_KM:.0f} km Abar={st['Abar']:.4f}")
        rows.append({"id": i, "file": fname, "Ye0": Ye, "s0": s0,
                     "v_inf_c": v, "tau_s": tau, "rho0_gcc": rho0,
                     "Abar": st["Abar"], "Zbar": st["Zbar"],
                     "Xn": st["Xn"], "Xp": st["Xp"], "Xalpha": st["Xa"],
                     "Xheavy": st["Xheavy"], "s_ion": st["s_ion"],
                     "s_ele": st["s_ele"], "s_rad": st["s_rad"],
                     "flag_rho0_gt_1e12": int(rho0 > RHO0_FLAG)})
        if i % 128 == 0:
            print(f"  {i}/{len(P)}  ({time.time() - t_start:.0f} s)")

    with open(out / "trajectories_index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    nflag = sum(r["flag_rho0_gt_1e12"] for r in rows)
    print(f"wrote {len(rows)} trajectories to {out}  "
          f"({nflag} with rho0 > 1e12 g/cm3)")


if __name__ == "__main__":
    main()

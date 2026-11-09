"""
initial_state.py
================

Initial thermodynamic state of an ejecta fluid element at NSE freeze-out:
given (T0, s0, Ye0) find the density rho0 and the NSE composition such that

    s_tot(T0, rho0, Ye0) = s_ion(NSE) + s_e+-(T0, rho0*Ye0) + s_gamma(T0, rho0)
                         = s0                                 [k_B / baryon]

* NSE composition: solved here with WinNet's own nuclear data
  (Nuclear_data/winvne_v2.0.dat: A, Z, N, spin J, mass excess and the
  normalized partition functions G(T) on WinNet's 24-point T9 grid), so the
  initial composition is the one WinNet itself would find.  Coulomb
  corrections to NSE are not included.
* s_ion: summed per species (Maxwell-Boltzmann, with spin degeneracy,
  partition function and its internal-excitation term), not the single
  (Abar, Zbar) ideal gas of the Helmholtz ion term.
* s_e+- and s_gamma: Helmholtz EOS (Timmes & Swesty 2000; the electron-
  positron physics is that of Timmes & Arnett 1999), via HelmholtzEOS/run_helm.exe.

Units: cgs, temperatures in K unless named T9; entropies in k_B per baryon.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
WINVNE = HERE.parent / "Nuclear_data" / "winvne_v2.0.dat"
HELM_DIR = HERE / "HelmholtzEOS"
HELM_EXE = HELM_DIR / "run_helm.exe"

# constants (CODATA 2018), cgs
K_B = 1.380649e-16            # erg/K
N_A = 6.02214076e23           # 1/mol
M_U = 1.66053906660e-24       # g
HBAR = 1.054571817e-27        # erg s
MEV = 1.602176634e-6          # erg
GK = 1.0e9
NAK = N_A * K_B               # erg/g/K per (k_B/baryon)

T9_GRID = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                    1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0])


# ============================================================================
# 1. Nuclear data
# ============================================================================

@dataclass
class NucData:
    name: np.ndarray
    A: np.ndarray
    Z: np.ndarray
    N: np.ndarray
    J: np.ndarray
    dm: np.ndarray       # mass excess [MeV]
    G: np.ndarray        # normalized partition function, shape (nnuc, 24)


@lru_cache(maxsize=None)
def load_winvne(path: str = str(WINVNE)) -> NucData:
    """Parse winvne_v2.0.dat (header, name list, then 4-line blocks)."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    # data blocks start at the first line with 7 fields (name A Z N J dm ref)
    i0 = next(i for i, l in enumerate(lines) if len(l.split()) == 7)
    rec = {}                       # dict: a repeated name keeps its last block
    i = i0
    while i < len(lines):
        p = lines[i].split()
        if len(p) != 7:
            i += 1
            continue
        g = np.array(" ".join(lines[i + 1:i + 4]).split(), dtype=float)
        rec[p[0]] = (float(p[1]), int(p[2]), int(p[3]), float(p[4]),
                     float(p[5]), g)
        i += 4
    names = np.array(list(rec))
    v = list(rec.values())
    return NucData(name=names,
                   A=np.array([r[0] for r in v]),
                   Z=np.array([r[1] for r in v], dtype=float),
                   N=np.array([r[2] for r in v], dtype=float),
                   J=np.array([r[3] for r in v]),
                   dm=np.array([r[4] for r in v]),
                   G=np.array([r[5] for r in v]))


def _lnG_and_dlnG(nuc: NucData, T9: float):
    """ln G(T) and d ln G / d ln T, by cubic spline in (ln T9, ln G)."""
    lnG = np.log(nuc.G)
    sp = CubicSpline(np.log(T9_GRID), lnG, axis=1)
    x = np.log(T9)
    return sp(x), sp(x, 1)


# ============================================================================
# 2. NSE solver
# ============================================================================

class NSE:
    """NSE at fixed temperature for all winvne nuclei.

        Y_i = g_i G_i(T) n_Q,i / (rho N_A) * exp(Z_i eta_p + N_i eta_n + B_i/kT)

    eta_{n,p} = mu_{n,p}/kT (non-relativistic, rest mass removed);
    B_i = Z dm_p + N dm_n - dm_i;  n_Q,i = (A_i m_u kT / 2 pi hbar^2)^(3/2).
    """

    def __init__(self, T: float, nuc: NucData | None = None):
        self.nuc = nuc or load_winvne()
        self.T = T
        kT = K_B * T
        n = self.nuc
        ip = np.where(n.name == "p")[0][0]
        ineu = np.where(n.name == "n")[0][0]
        B = n.Z * n.dm[ip] + n.N * n.dm[ineu] - n.dm          # MeV
        lnG, self.dlnG = _lnG_and_dlnG(n, T / GK)
        nQ = (n.A * M_U * kT / (2 * np.pi * HBAR ** 2)) ** 1.5
        # ln Y_i = c_i - ln(rho) + Z eta_p + N eta_n
        self.B_kT = B * MEV / kT
        self.c = np.log(2 * n.J + 1) + lnG + np.log(nQ / N_A) + self.B_kT
        self.ineu, self.ip = ineu, ip

    def lnY(self, rho, eta_n, eta_p):
        n = self.nuc
        return self.c - np.log(rho) + n.Z * eta_p + n.N * eta_n

    def solve(self, rho: float, Ye: float, guess=None, tol=1e-11, maxit=200):
        """Return (eta_n, eta_p, Y) with sum X = 1 and sum Z Y = Ye."""
        n = self.nuc
        if guess is None:
            # free-nucleon guess
            guess = (np.log(1 - Ye) - self.c[self.ineu] + np.log(rho),
                     np.log(Ye) - self.c[self.ip] + np.log(rho))
        x = np.array(guess, float)
        lnYe = np.log(Ye)

        def resid(x):
            ly = self.lnY(rho, *x)
            m = ly.max()
            w = np.exp(ly - m)
            SA, SZ = np.dot(n.A, w), np.dot(n.Z, w)
            f = np.array([m + np.log(SA), m + np.log(SZ) - lnYe])
            # analytic Jacobian of (ln sum A Y, ln sum Z Y) wrt (eta_n, eta_p)
            J = np.array([[np.dot(n.A * n.N, w) / SA, np.dot(n.A * n.Z, w) / SA],
                          [np.dot(n.Z * n.N, w) / SZ, np.dot(n.Z * n.Z, w) / SZ]])
            return f, J

        f, J = resid(x)
        for _ in range(maxit):
            if np.max(np.abs(f)) < tol:
                break
            dx = np.linalg.solve(J, -f)
            # backtracking line search on |f|
            lam, f0 = 1.0, np.linalg.norm(f)
            while True:
                xn = x + lam * dx
                fn, Jn = resid(xn)
                if np.linalg.norm(fn) < (1 - 1e-4 * lam) * f0 or lam < 1e-6:
                    break
                lam *= 0.5
            x, f, J = xn, fn, Jn
        else:
            raise RuntimeError(f"NSE not converged: T={self.T:.3e} rho={rho:.3e} "
                               f"Ye={Ye:.4f} |f|={np.max(np.abs(f)):.2e}")
        Y = np.exp(self.lnY(rho, *x))
        return x[0], x[1], Y

    def s_ion(self, rho, Ye, guess=None):
        """Ion entropy per baryon of the NSE mixture, and the solution.

        s_ion = sum_i Y_i [5/2 + ln(g G n_Q/(n_i)) + dlnG/dlnT]
              = sum_i Y_i [5/2 - (Z eta_p + N eta_n + B/kT) + dlnG/dlnT]
        evaluated directly from ln Y_i for numerical robustness.
        """
        eta_n, eta_p, Y = self.solve(rho, Ye, guess)
        n = self.nuc
        # ln(g G n_Q / n_i) = -(Z eta_p + N eta_n) - B/kT
        ln_q_over_n = -(n.Z * eta_p + n.N * eta_n) - self.B_kT
        s = np.sum(Y * (2.5 + ln_q_over_n + self.dlnG))
        return s, (eta_n, eta_p), Y


# ============================================================================
# 3. Electron/positron + radiation entropy (Helmholtz EOS)
# ============================================================================

def helm_entropy(T, rho, Ye):
    """Arrays (s_e+-, s_rad) in k_B/baryon from HelmholtzEOS/run_helm.exe."""
    T, rho, Ye = np.broadcast_arrays(np.asarray(T, float),
                                     np.asarray(rho, float),
                                     np.asarray(Ye, float))
    inp = f"{T.size}\n" + "\n".join(
        f"{a:.15e} {b:.15e} {c:.15e}" for a, b, c in
        zip(T.ravel(), rho.ravel(), Ye.ravel())) + "\n"
    out = subprocess.run([str(HELM_EXE)], input=inp, capture_output=True,
                         text=True, cwd=HELM_DIR, check=True).stdout
    a = np.array(out.split(), float).reshape(-1, 2) / NAK
    return a[:, 0].reshape(T.shape), a[:, 1].reshape(T.shape)


class LeptonPhotonEntropy:
    """s_e+- and s_rad at fixed T, tabulated once and splined.

    s_e+- per baryon depends on (T, rho*Ye) only and scales as
    Ye * h(rho*Ye); s_rad = 4 a T^3 / (3 rho N_A k_B) is exact.
    """

    def __init__(self, T, lgx=(-2.0, 15.0), npts=1701):
        self.T = T
        x = np.logspace(*lgx, npts)                  # x = rho*Ye
        se, sr = helm_entropy(T, x, 1.0)
        self._h = CubicSpline(np.log(x), np.log(se))
        self._rad1 = sr[0] * x[0]                    # s_rad * rho (const)

    def __call__(self, rho, Ye):
        se = Ye * np.exp(self._h(np.log(rho * Ye)))
        return se, self._rad1 / rho


# ============================================================================
# 4. rho0(T0, s0, Ye0)
# ============================================================================

class InitialStateSolver:
    def __init__(self, T0=7.0 * GK):
        self.T0 = T0
        self.nse = NSE(T0)
        self.lep = LeptonPhotonEntropy(T0)
        self._last = None

    def s_tot(self, rho, Ye, parts=False):
        si, guess, Y = self.nse.s_ion(rho, Ye, self._last)
        self._last = guess
        se, sr = self.lep(rho, Ye)
        return (si, se, sr, Y) if parts else si + se + sr

    def entropy_curve(self, Ye, lgrho=(2.0, 13.0), dlg=0.25):
        """s_tot(log10 rho) at fixed (T0, Ye) on a grid, walking upward in
        density and warm-starting each NSE solve from the previous one
        (a cold free-nucleon guess fails at rho >~ 1e12)."""
        lrs = np.arange(lgrho[0], lgrho[1] + 1e-9, dlg)
        self._last = None
        s, guesses = [], []
        for lr in lrs:
            s.append(self.s_tot(10.0 ** lr, Ye))
            guesses.append(self._last)
        return lrs, np.array(s), guesses

    def rho0(self, s0, Ye, lgrho=(2.0, 13.0), curve=None):
        """Solve s_tot(T0, rho, Ye) = s0 for rho (Brent in log10 rho).

        `curve` = output of entropy_curve(Ye) can be passed to reuse the
        bracketing scan across many s0 at the same Ye.
        """
        lrs, s, guesses = curve or self.entropy_curve(Ye, lgrho)
        if not (s[-1] < s0 < s[0]):
            raise ValueError(f"s0={s0} outside [{s[-1]:.3g}, {s[0]:.3g}] "
                             f"for Ye={Ye} on rho in 1e{lgrho[0]}-1e{lgrho[1]}")
        k = np.where(s < s0)[0][0]               # s decreases with rho
        g0 = guesses[k - 1]

        def f(lr):
            self._last = g0                      # warm start from the bracket
            return self.s_tot(10.0 ** lr, Ye) - s0

        lr = brentq(f, lrs[k - 1], lrs[k], xtol=1e-8)
        self._last = g0
        rho = 10.0 ** lr
        si, se, sr, Y = self.s_tot(rho, Ye, parts=True)
        n = self.nse.nuc
        abar = 1.0 / Y.sum()
        return {"rho0": rho, "s_ion": si, "s_ele": se, "s_rad": sr,
                "Abar": abar, "Zbar": abar * np.dot(n.Z, Y),
                "Xn": Y[self.nse.ineu], "Xp": Y[self.nse.ip],
                "Xa": 4 * Y[n.name == "he4"][0],
                "Xheavy": float(np.sum((n.A * Y)[n.Z > 2])),
                "Y": Y}

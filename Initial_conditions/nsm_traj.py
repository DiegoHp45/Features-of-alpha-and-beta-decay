"""
nsm_traj.py
===========

Parameterized thermodynamic trajectories for r-process nucleosynthesis in
neutron-star-merger ejecta.

Design philosophy
-----------------
The nucleosynthesis is controlled by the state of the fluid element at the
moment NSE freezes out (T0 ~ 6-10 GK) plus the rate at which the density
subsequently drops.  Everything earlier is erased by NSE.  The model therefore
carries exactly three free parameters per trajectory,

    theta = (Ye0, s0, tau)

evaluated at T = T0, plus two optional structural parameters:

    n     : deceleration/geometry index of the density law (n = 3 -> spherical
            homologous; n -> inf -> pure exponential)
    t_hom : time at which the profile is forced onto rho ~ t^-3

and, if neutrino irradiation is switched on, (L_nue, L_nuebar, E_nue, E_nuebar)
plus the radius history r(t).

References for the functional forms
-----------------------------------
LR15   Lippuner & Roberts 2015, ApJ 815, 82   (https://doi.org/10.1088/0004-637X/815/2/82)  -- Eq. (1)  rho(t)
K12    Korobkin et al. 2012, MNRAS 426, 1940  (arXiv:1206.2379)   -- Sec. 3.1
R18    Radice et al. 2018, ApJ 869, 130       (https://doi.org/10.3847/1538-4357/aaf054)    -- tau def.
KAR25  Kuske, Arcones & Reichert 2025, ApJ    (https://doi.org/10.3847/1538-4357/adf0f7)    -- Eqs. (1),(7)
HWQ97  Hoffman, Woosley & Qian 1997, ApJ 482, 951                 -- s^3/tau
QW96   Qian & Woosley 1996, ApJ 471, 331                          -- nu rates
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
from dataclasses import dataclass, field
from scipy import optimize

# ----------------------------------------------------------------------------
# physical constants (cgs unless stated)
# ----------------------------------------------------------------------------
C_LIGHT = 2.99792458e10          # cm / s
K_B = 1.380649e-16               # erg / K
MEV = 1.602176634e-6             # erg
N_A = 6.02214076e23              # 1 / g
A_RAD = 7.5657e-15               # erg cm^-3 K^-4
M_U = 1.66053907e-24             # g
GK = 1.0e9                       # K
SIGMA_0 = 1.76e-44               # cm^2
G_A = 1.2723
DELTA_NP = 1.2933                # MeV, m_n - m_p
M_E_C2 = 0.51100                 # MeV
h = 6.62607015e-27               # erg s

# ============================================================================
# 1. DENSITY PARAMETERIZATIONS
# ============================================================================

def rho_lr15(t, rho0, tau):
    """Lippuner & Roberts (2015) Eq. (1); also KAR25 Eq. (1).

    rho = rho0 exp(-t/tau)          t <= 3 tau
    rho = rho0 (3 tau / (e t))^3     t >= 3 tau

    C^1 continuous at t = 3 tau by construction.  tau is the e-folding time of
    the density at t = 0, i.e. tau = -(d ln rho / dt)^-1 |_{t=0}.
    """
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    early = t <= 3.0 * tau
    out[early] = rho0 * np.exp(-t[early] / tau)
    late = ~early
    out[late] = rho0 * (3.0 * tau / (np.e * t[late])) ** 3
    return out


def rho_powerlaw(t, rho0, tau, n=3.0):
    """Single-branch generalized homologous law.

        rho(t) = rho0 [1 + t/(n tau)]^{-n}

    Properties
    ----------
    * -(d ln rho/dt)|_0 = 1/tau  for every n (same tau convention as LR15).
    * n = 3  : exact for a uniform sphere in free expansion, r(t) = r0(1 + t/3tau),
               v = r0/(3 tau) = const.  Asymptotes to rho ~ t^-3.
    * n = 2  : cylindrical (relevant for equatorial tidal tails before they
               become spherical).
    * n -> inf : recovers rho0 exp(-t/tau), i.e. the early LR15 branch.

    Unlike LR15 this has no kink and no e^3 ~ 20 amplitude offset between the
    early and late branches, which matters if you extrapolate to seconds.
    """
    t = np.asarray(t, dtype=float)
    return rho0 * (1.0 + t / (n * tau)) ** (-n)


def rho_freefall_homologous(t, rho_fin, t_fin):
    """Korobkin et al. (2012) extrapolation: rho = rho_fin (t/t_fin)^-3.

    This is the standard extrapolation used past the end of the hydro run,
    not a full-trajectory parameterization.  It is only valid once the fluid
    element has reached terminal velocity (see `homology_diagnostic`).
    """
    t = np.asarray(t, dtype=float)
    return rho_fin * (t / t_fin) ** (-3.0)


def rho_two_stage(t, rho0, tau, tau2, t_break, n=3.0):
    """Piecewise two-timescale profile for shocked / re-accelerated elements.

    Fast initial decompression (tau) followed by a slower phase (tau2) after
    t_break -- e.g. an element that is shocked, decompresses, then coasts in a
    slowly expanding tail, or a disk outflow that stalls.  Continuity in rho is
    enforced; the derivative is intentionally discontinuous, since the physical
    situation it models (a shock) is too.
    """
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    a = t <= t_break
    out[a] = rho_powerlaw(t[a], rho0, tau, n)
    rho_b = rho_powerlaw(np.array([t_break]), rho0, tau, n)[0]
    out[~a] = rho_powerlaw(t[~a] - t_break, rho_b, tau2, n)
    return out


DENSITY_MODELS = {
    "lr15": lambda t, p: rho_lr15(t, p["rho0"], p["tau"]),
    "powerlaw": lambda t, p: rho_powerlaw(t, p["rho0"], p["tau"], p.get("n", 3.0)),
    "two_stage": lambda t, p: rho_two_stage(
        t, p["rho0"], p["tau"], p["tau2"], p["t_break"], p.get("n", 3.0)),
}


# ----------------------------------------------------------------------------
# 1b. TEMPERATURE FROM DENSITY: T ~ rho^{1/3}
# ----------------------------------------------------------------------------

def T_from_rho(rho, rho0, T0):
    """Temperature history implied by adiabatic (constant-entropy) expansion.

    In the radiation/NSE-dominated regime entropy per baryon scales as
    s ~ T^3 / rho, so holding s fixed as the fluid element decompresses gives

        T(t) = T0 * (rho(t) / rho0)^(1/3)

    i.e. the temperature tracks the cube root of the density, normalized so
    that T(t=0) = T0 when rho(t=0) = rho0.  This replaces the previous
    approximation of freezing T at T0 for the whole trajectory.
    """
    rho = np.asarray(rho, dtype=float)
    return T0 * (rho / rho0) ** (1.0 / 3.0)


# ============================================================================
# 2. INITIAL DENSITY FROM (s0, Ye0, T0)
# ============================================================================


def rho0_from_entropy(s0, T0=7.0 * GK, Abar=None):
    return np.exp(5/2-s0*Abar)*((2*np.pi*M_U)**(3/2))*((K_B*T0)**(3/2))*(Abar**(5/2))/(N_A*h**3)

# ----------------------------------------------------------------------------
# 2b. NSE composition (real Abar/Zbar, replacing the crude guess above)
# ----------------------------------------------------------------------------
# `simple/public_nse.f90` (Timmes' public_nsm distribution) is a self-contained
# 47-isotope NSE solver: given (T, rho, Ye) it root-finds the neutron/proton
# chemical potentials and returns the NSE mass-fraction vector.  It ships as
# an interactive REPL, which is awkward to script against, so
# `simple/run_nse.f90` wraps the *same* subroutines (built via
# `simple/Makefile`, which extracts them out of public_nse.f90 so there is
# only one copy of the physics) into a single-shot command-line driver that
# prints a fixed, parseable block including Abar/Zbar.

_NSE_DIR = Path(__file__).resolve().parent / "simple"
_NSE_EXE = _NSE_DIR / "run_nse.exe"
_NSE_T_FLOOR = 2.0 * GK          # hard floor baked into public_nse.f90's nse()
_nse_built = False


def _ensure_nse_built():
    """Build simple/run_nse.exe (via simple/Makefile) if not already built."""
    global _nse_built
    if _nse_built and _NSE_EXE.exists():
        return
    if not _NSE_EXE.exists():
        if shutil.which("make") is None or shutil.which("gfortran") is None:
            raise RuntimeError(
                f"{_NSE_EXE} not found, and `make`/`gfortran` are not on "
                f"PATH to build it. Build it yourself: "
                f"cd {_NSE_DIR} && make run_nse.exe"
            )
        result = subprocess.run(["make", "run_nse.exe"], cwd=_NSE_DIR,
                                 capture_output=True, text=True)
        if result.returncode != 0 or not _NSE_EXE.exists():
            raise RuntimeError(
                f"failed to build {_NSE_EXE}:\n{result.stdout}\n{result.stderr}")
    _nse_built = True


def run_nse(T, rho, Ye, timeout=180.0):
    """Call the compiled NSE solver at one (T, rho, Ye) point.

    T [K], rho [g/cm^3], Ye [dimensionless].  Requires T > 2.0e9 K -- below
    that public_nse.f90 itself refuses (its `nse` routine prints
    'no nse state'), matching the physical NSE breakdown temperature.

    Returns
    -------
    dict with keys:
      'xmass'    : {isotope_name: mass_fraction}
      'aion'     : {isotope_name: A}
      'zion'     : {isotope_name: Z}
      'abar'     : mean nucleon number of the NSE mixture, 1/Abar = sum X_i/A_i
      'zbar'     : mean charge, Zbar = Abar * sum X_i Z_i/A_i
      'ye_check' : Zbar/Abar -- should reproduce the input Ye to solver tolerance
    """
    if T <= _NSE_T_FLOOR:
        raise ValueError(
            f"T={T:.3e} K is below the NSE validity floor "
            f"({_NSE_T_FLOOR:.1e} K, baked into public_nse.f90); NSE is not "
            "a physical description down there -- use a different "
            "composition treatment for this trajectory segment."
        )
    _ensure_nse_built()
    proc = subprocess.run(
        [str(_NSE_EXE), f"{T:.15e}", f"{rho:.15e}", f"{Ye:.15e}"],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"run_nse.exe failed (exit {proc.returncode}) at "
            f"T={T:.3e} rho={rho:.3e} Ye={Ye:.3f}:\n{proc.stderr}"
        )
    xmass, aion, zion = {}, {}, {}
    abar = zbar = ye_check = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "ABAR":
            abar = float(parts[1])
        elif parts[0] == "ZBAR":
            zbar = float(parts[1])
        elif parts[0] == "YE_CHECK":
            ye_check = float(parts[1])
        elif len(parts) == 4:
            name, a, z, x = parts
            aion[name] = float(a)
            zion[name] = float(z)
            xmass[name] = float(x)
    if abar is None or zbar is None:
        raise RuntimeError(
            f"could not parse ABAR/ZBAR from run_nse.exe output:\n{proc.stdout}")
    return {"xmass": xmass, "aion": aion, "zion": zion,
            "abar": abar, "zbar": zbar, "ye_check": ye_check}


def rho0_and_nse_state(s0, Ye0, T0=7.0 * GK, max_iter=6, tol_dex=1e-3):
    """Self-consistently solve for rho0 and the NSE composition at (s0, Ye0, T0).

    `rho0_from_entropy` needs an Abar to evaluate the ion entropy term, and
    Abar is exactly what NSE gives you -- so this iterates: get rho0 from a
    guessed/previous Abar, call the real NSE solver at that (T0, rho0, Ye0)
    to get a better Abar, refine rho0, and repeat until rho0 stops moving by
    more than `tol_dex` in log10.  This is the "real NSE call" the docstrings
    above say to substitute for the crude Abar guess.

    Returns a dict: rho0, abar, zbar, xmass (NSE mass-fraction dict), aion,
    zion, ye_check, n_iter.
    """
    rho0 = 1e10#rho0_from_entropy(s0, T0, Abar=2)  # initial guess for Abar)
    nse_state, it = None, 0
    for it in range(1, max_iter + 1):
        nse_state = run_nse(T0, rho0, Ye0)
        rho0_new = rho0_from_entropy(s0, T0, Abar=nse_state["abar"])
        converged = abs(np.log10(rho0_new) - np.log10(rho0)) < tol_dex
        rho0 = rho0_new
        if converged:
            break
    else:
        warnings.warn(
            f"rho0<->NSE(Abar) loop did not converge in {max_iter} "
            f"iterations (s0={s0}, Ye0={Ye0}, T0={T0:.3e})"
        )
    return {"rho0": rho0, "n_iter": it, **nse_state}


# ============================================================================
# 3. KINEMATICS AND CONSISTENCY DIAGNOSTICS
# ============================================================================

def radius_from_rho(t, rho, r0, n=3.0):
    """Radius history consistent with baryon-number conservation for a
    self-similar element: rho r^n = const  =>  r(t) = r0 (rho0/rho)^{1/n}.

    Needed for neutrino irradiation (flux ~ 1/r^2) and for the causality check.
    """
    rho = np.asarray(rho, dtype=float)
    return r0 * (rho[0] / rho) ** (1.0 / n)


def homology_diagnostic(t, rho, r0, n=3.0, r_measured=None):
    """Return (v(t), dlnv/dlnt, t_hom) where t_hom is the time after which
    |d ln v / d ln t| < 0.05, i.e. the onset of genuine homology.

    Physical check: free expansion (rho ~ t^-3) is only legitimate once the
    specific enthalpy excess has been converted to kinetic energy,
        (h - 1) c^2  <<  v^2 / 2 .
    For s ~ 10-30 k_B/baryon and T ~ 3 GK the thermal energy is ~ few MeV per
    baryon while v = 0.2 c gives ~ 19 MeV per baryon, so homology is reached
    within a few ms -- but this must be verified per trajectory, not assumed.

    IMPORTANT: pass `r_measured` = the tracer's actual r(t) when testing a
    *simulation* trajectory.  If you let the radius be reconstructed from rho
    via baryon conservation, the test is circular: the reconstruction already
    assumes the self-similar law whose validity you are trying to check.  On a
    parameterized trajectory this function therefore only verifies internal
    consistency and the causality bound, not homology itself.
    """
    r = np.asarray(r_measured, float) if r_measured is not None \
        else radius_from_rho(t, rho, r0, n)
    v = np.gradient(r, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlnv = np.gradient(np.log(np.abs(v) + 1e-300), np.log(t + 1e-300))
    idx = np.where(np.abs(dlnv) < 0.05)[0]
    t_hom = t[idx[0]] if len(idx) else np.nan
    return v, dlnv, t_hom


def causality_ok(v, v_max=0.9 * C_LIGHT):
    """Hard filter: reject trajectories whose implied velocity is superluminal
    or unphysically fast.  Extrapolating the LR15 law to late times without
    this check produces v > c (LR15 note this explicitly in their Sec. 3.1)."""
    return np.all(np.abs(v) < v_max)



# ============================================================================
# 4. CALIBRATION AGAINST TRACER DATA
# ============================================================================
'''
def fit_tau(t_trace, rho_trace, model="lr15", t0=None, n_fixed=3.0,
            fit_n=False, weights=None, rho_floor=1e2):
    """Fit a density parameterization to one tracer, in log-density space.

    Fitting in log rho is essential: the density drops by 8-12 decades and a
    linear-space L2 loss would be dominated entirely by the first few points.

    Parameters
    ----------
    t_trace, rho_trace : arrays, tracer time [s] and rest-mass density [g/cm^3]
    t0                 : time origin.  If None, the *last* time the tracer is
                         above T0 should be passed in; the trajectory is
                         re-zeroed there so that tau has the LR15 meaning.
    fit_n              : if True, also fit the deceleration index n (only for
                         model='powerlaw').

    Returns
    -------
    dict with keys rho0, tau, (n), plus 'rms_dex' = RMS residual in dex and
    'cost'.  Use rms_dex as the primary goodness-of-fit statistic; < 0.1 dex
    is an excellent match, > 0.3 dex usually signals a shock (non-monotonic
    rho) that no smooth parameterization can capture (KAR25 Fig. 11).
    """
    t = np.asarray(t_trace, float)
    rho = np.asarray(rho_trace, float)
    if t0 is not None:
        m = t >= t0
        t, rho = t[m] - t0, rho[m]
    good = rho > rho_floor
    t, rho = t[good], rho[good]
    y = np.log10(rho)
    w = np.ones_like(y) if weights is None else np.asarray(weights, float)[good]

    rho0_guess = rho[0]
    # tau guess from the initial logarithmic derivative
    if len(t) > 3:
        slope = np.polyfit(t[:max(3, len(t) // 10)],
                           np.log(rho[:max(3, len(t) // 10)]), 1)[0]
        tau_guess = -1.0 / slope if slope < 0 else 1e-3
    else:
        tau_guess = 1e-3

    def resid(p):
        d = {"rho0": 10.0 ** p[0], "tau": np.exp(p[1])}
        if model == "powerlaw":
            d["n"] = np.exp(p[2]) if fit_n else n_fixed
        return w * (np.log10(DENSITY_MODELS[model](t, d)) - y)

    p0 = [np.log10(rho0_guess), np.log(max(tau_guess, 1e-6))]
    if model == "powerlaw" and fit_n:
        p0.append(np.log(n_fixed))
    sol = least_squares(resid, p0, loss="soft_l1", f_scale=0.2)

    out = {"rho0": 10.0 ** sol.x[0], "tau": float(np.exp(sol.x[1]))}
    if model == "powerlaw":
        out["n"] = float(np.exp(sol.x[2])) if fit_n else n_fixed
    out["rms_dex"] = float(np.sqrt(np.mean(resid(sol.x) ** 2)))
    out["cost"] = float(sol.cost)
    out["n_points"] = int(len(t))
    return out


def state_at_T0(t_trace, T_trace, s_trace, ye_trace, rho_trace, T0=7.0 * GK):
    """Interpolate (t0, rho0, s0, Ye0) at the *last* downward crossing of T0.

    'Last' matters: shock-reheated elements cross T0 several times, and the
    physically relevant NSE exit is the final one (KAR25 Sec. 4).  Returns
    None if the tracer never gets above T0 (a genuinely cold tidal element --
    for those, NSE was never re-established and the initial composition must
    come from the cold NS, not from NSE; see K12 discussion).
    """
    T = np.asarray(T_trace, float)
    t = np.asarray(t_trace, float)
    above = T > T0
    if not above.any():
        return None
    i = np.where(above)[0][-1]
    if i + 1 >= len(T):
        return None
    f = (T[i] - T0) / (T[i] - T[i + 1])
    lin = lambda a: a[i] + f * (a[i + 1] - a[i])
    return {"t0": float(lin(t)), "rho0": float(np.exp(lin(np.log(rho_trace)))),
            "s0": float(lin(np.asarray(s_trace, float))),
            "Ye0": float(lin(np.asarray(ye_trace, float))),
            "n_crossings": int(np.sum(np.diff(above.astype(int)) != 0))}

'''
def abundance_distance(X1, X2, A_max=250, floor=1e-7):
    """KAR25 Eq. (7): mean |log X| difference over mass number.

        d_ij = (1/A_max) sum_A |log X_i(A) - log X_j(A)|

    X1, X2 are arrays of mass fractions indexed by A (1..A_max).  Only bins
    where both exceed `floor` are counted.  Benchmarks from KAR25: the mean
    tracer-vs-model distance over ~20 hydro models is d = 0.23 (+0.15 -0.18),
    i.e. a typical agreement of a quarter dex per mass bin.  Treat d < 0.3 as
    a successful reproduction, d > 0.5 as a failure needing investigation.
    """
    X1 = np.asarray(X1, float)[:A_max]
    X2 = np.asarray(X2, float)[:A_max]
    m = (X1 > floor) & (X2 > floor)
    if not m.any():
        return np.inf
    return float(np.mean(np.abs(np.log10(X1[m]) - np.log10(X2[m])))) 


def mass_weighted_histogram(values, masses, bins):
    """Mass-weighted 1D histogram -- the correct way to compare a tracer
    ensemble to a parameterized grid.  Individual trajectories are not
    comparable one-to-one; only the mass-weighted *distributions* of
    (Ye0, s0, tau) and of the final Y(A) are."""
    h, edges = np.histogram(values, bins=bins, weights=masses)
    return h / np.sum(masses), edges


# ============================================================================
# 5. NEUTRINO IRRADIATION (optional)
# ============================================================================
'''
def sigma_nue_n(E_nu_MeV):
    """nu_e + n -> p + e^-  cross section, allowed approximation.
        sigma = (1 + 3 g_A^2)/4 * sigma_0 * ((E + Delta)/m_e c^2)^2
    ~ 9.6e-44 cm^2 (E/MeV)^2 for E >> Delta."""
    return 0.25 * (1.0 + 3.0 * G_A ** 2) * SIGMA_0 * \
        ((E_nu_MeV + DELTA_NP) / M_E_C2) ** 2


def sigma_nuebar_p(E_nu_MeV):
    """nubar_e + p -> n + e^+ (threshold Delta + m_e ignored above ~2 MeV)."""
    return 0.25 * (1.0 + 3.0 * G_A ** 2) * SIGMA_0 * \
        (np.maximum(E_nu_MeV - DELTA_NP, 0.0) / M_E_C2) ** 2


def nu_capture_rates(r_cm, L_nue, L_nuebar, E_nue, E_nuebar,
                     eps_ratio=1.2, R_nu=3.0e6):
    """Per-nucleon neutrino capture rates [1/s].

        lambda = (L_nu / (4 pi r^2 <E_nu>)) * <sigma> * Phi_geo

    L in erg/s, <E> in MeV.  `eps_ratio` = <E^2>/<E>^2 for the assumed spectrum
    (1.2 for a Fermi-Dirac with eta ~ 3); the capture cross sections scale as
    E^2, so <sigma> ~ sigma(<E>) * eps_ratio.  Phi_geo is the flux-dilution
    correction for a finite neutrinosphere of radius R_nu.

    Typical BNS remnant values within the first ~10 ms (Cusinato et al. 2022;
    Perego et al. 2017; Rosswog & Liebendoerfer 2003):
        L_nue ~ 0.5-2e53, L_nuebar ~ 1-4e53 erg/s  (L_nuebar / L_nue ~ 2-3)
        <E_nue> ~ 8-12 MeV, <E_nuebar> ~ 13-19 MeV, <E_nux> ~ 18-25 MeV
    """
    r = np.asarray(r_cm, float)
    phi = 0.5 * (1.0 - np.sqrt(np.maximum(1.0 - (R_nu / np.maximum(r, R_nu)) ** 2, 0.0)))
    phi = np.maximum(phi, 1e-12) * 2.0  # normalized so phi -> (R_nu/2r)^2 far away
    flux_nue = L_nue / (4 * np.pi * r ** 2 * E_nue * MEV)
    flux_nueb = L_nuebar / (4 * np.pi * r ** 2 * E_nuebar * MEV)
    lam_nue = flux_nue * sigma_nue_n(E_nue) * eps_ratio
    lam_nueb = flux_nueb * sigma_nuebar_p(E_nuebar) * eps_ratio
    return lam_nue, lam_nueb


def ye_equilibrium(lam_nue, lam_nuebar):
    """Weak (neutrino-only) equilibrium electron fraction,
        Ye_eq = lambda_nue / (lambda_nue + lambda_nuebar),
    which is *independent of radius* because both rates scale as 1/r^2.  This
    is why an irradiated outflow forgets its initial Ye and why Ye0 cannot be
    varied independently of (L_nue/L_nuebar, <E_nue>, <E_nuebar>) once the
    exposure integral exceeds unity.
    """
    return lam_nue / (lam_nue + lam_nuebar)


def nu_exposure(t, lam_nue, lam_nuebar):
    """Dimensionless exposure X = int (lam_nue + lam_nuebar) dt.
    X << 1  -> Ye frozen at its hydro value (tidal ejecta: valid).
    X >~ 1  -> Ye driven to Ye_eq (neutrino-driven wind: valid).
    0.1 < X < 10 is the awkward regime where Ye0 is genuinely a free parameter
    only in the sense of 'unknown', not 'independent'."""
    return np.trapezoid(np.asarray(lam_nue) + np.asarray(lam_nuebar), t)

'''
# ============================================================================
# 6. PHYSICAL-ADMISSIBILITY FILTERS FOR THE PARAMETER SPACE
# ============================================================================

@dataclass
class ParameterBox:
    """Prior ranges for (Ye0, s0, tau) by ejecta component.

    Defaults are the mass-weighted ranges reported in the literature; see
    KAR25 Table 3 for the tracer-derived means/1-sigma per hydro model.
    """
    ye: tuple = (0.01, 0.5)
    s: tuple = (1.0, 290.0)          # k_B / baryon
    tau: tuple = (0.25e-3, 600e-3)   # s

    @staticmethod
    def tidal():
        return ParameterBox(ye=(0.01, 0.20), s=(1, 10.0), tau=(0.2e-3, 2e-3))

    @staticmethod
    def shocked_dynamical():
        return ParameterBox(ye=(0.05, 0.45), s=(5.0, 150.0), tau=(0.5e-3, 10e-3))

    @staticmethod
    def nu_driven_wind():
        return ParameterBox(ye=(0.25, 0.45), s=(10.0, 30.0), tau=(5e-3, 30e-3))

    @staticmethod
    def disk_wind():
        return ParameterBox(ye=(0.10, 0.40), s=(8.0, 40.0), tau=(20e-3, 200e-3))

'''
def neutron_to_seed_proxy(s0, tau, Ye0):
    """Scaling proxy for Y_n / Y_seed (HWQ97).

    Seed nuclei are assembled by the alpha-process, whose rate is dominated by
    the (effectively) three-body bottleneck, so d Y_seed/dt ~ rho^2.  At fixed
    temperature and in the radiation-dominated regime rho ~ T^3 / s, hence

        Y_seed ~ rho^2 tau ~ tau / s^3   =>   Y_n / Y_seed ~ (1 - 2 Ye) s^3 / tau.

    Use this only to *order* the parameter space and to define iso-nucleosynthesis
    surfaces; it fails badly in the ion-dominated corner (s < 30 k_B/baryon and
    Ye < 0.35), where NSE already contains heavy nuclei and Y_seed is set by the
    EOS rather than by the alpha-process (KAR25 Fig. 8).
    """
    return np.maximum(1.0 - 2.0 * np.asarray(Ye0, float), 1e-6) * \
        np.asarray(s0, float) ** 3 / np.asarray(tau, float)
'''

def admissible(Ye0, s0, tau, rho0, r0, v_term_max=0.8 * C_LIGHT,
               rho0_max=1e12, tau_min=0.2e-3):
    """Boolean admissibility mask encoding the hard physical constraints.

    1. Causality / energetics: v = r0 / (3 tau) must be sub-luminal and
       consistent with the Bernoulli-unbound velocities seen in GRHD
       (v_inf <~ 0.3 c for the bulk, tail to 0.6-0.8 c).
    2. EOS validity: rho0 <~ 1e12 g/cm^3, above which the Timmes EOS omits
       nuclear contributions and WinNet's heating treatment is switched off
       anyway (KAR25 Sec. 2.2).
    3. Numerical-resolution floor on tau: GRHD tracer sets do not resolve
       tau < ~0.5 ms (Radice et al. 2016 explicitly report no material below
       this), so smaller tau is an extrapolation, not a calibration.
    """
    v = r0 / (3.0 * np.asarray(tau, float))
    ok = (v < v_term_max)
    ok &= (np.asarray(rho0, float) < rho0_max)
    ok &= (np.asarray(tau, float) > tau_min)
    ok &= (np.asarray(Ye0, float) > 0.0) & (np.asarray(Ye0, float) < 0.5)
    ok &= (np.asarray(s0, float) > 0.0)
    return ok


def sample_correlated(n, mean_log, cov_log, seed=0):
    """Draw (Ye0, log10 s0, log10 tau) from a multivariate lognormal fitted to
    the tracer ensemble.  Fit `mean_log`/`cov_log` with np.cov on the tracer
    values -- this preserves the s-Ye correlation (shock-heated material is
    both hotter and more weak-processed; Radice et al. 2016 Fig. 10) and the
    tau-s anticorrelation, which a naive independent grid destroys.
    """
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean_log, cov_log, size=n)

# ============================================================================
# 8. WINNET OUTPUT
# ============================================================================

def write_winnet_trajectory(path, t, T9, rho, Ye=None, r_Km=None,
                            L_nue=None, L_nuebar=None,
                            E_nue=None, E_nuebar=None, header_note=""):
    """Write a whitespace-separated trajectory file.

    Columns:  time[s]  temperature[GK]  density[g/cm^3]  radius[Km]   [Ye]
    

    WinNet's `trajectory_format` runtime parameter lets you declare exactly
    this column layout, e.g.

        trajectory_format = "time:s, radius:cm, temp:GK, dens:g/cm3, ye"

    Check the column-name tokens against your WinNet version's
    `parameter.md` / `read_trajectory` routine before running -- the accepted
    unit strings have changed between releases.

    If you use WinNet's new parametric mode (KAR25) you do NOT need this file
    at all: you supply (Ye0, s0, T0) and a rho(t) function directly, and let
    the EOS produce T(t) self-consistently with nuclear heating.  Writing a
    fixed T(t) here means the network cannot feed heating back into the
    temperature, which suppresses the well-known ~1 GK plateau (K12 Fig. 7).
    """
    t = np.asarray(t, float)
    ncol = 4 + (Ye is not None) + 4 * (L_nue is not None)
    cols = [t, 
            np.asarray(T9, float), 
            np.asarray(rho, float),
            np.asarray(r_Km if r_Km is not None else np.zeros_like(t), float)]
    names = ["time[s]", "T[GK]", "rho[g/cm3]","radius[Km]"]
    if Ye is not None:
        cols.append(np.asarray(Ye, float) * np.ones_like(t)); names.append("Ye")
    if L_nue is not None:
        for a, nm in ((L_nue, "Lnue[erg/s]"), (L_nuebar, "Lnuebar[erg/s]"),
                      (E_nue, "Enue[MeV]"), (E_nuebar, "Enuebar[MeV]")):
            cols.append(np.asarray(a, float) * np.ones_like(t)); names.append(nm)
    M = np.column_stack(cols)
    hdr = f"{header_note}\n" + "  ".join(names)
    np.savetxt(path, M, header=hdr, fmt="%18.10e")
    return path


# ============================================================================
# 9. END-TO-END DRIVER
# ============================================================================

@dataclass
class Trajectory:
    Ye0: float
    s0: float
    tau: float
    rho0: float
    n: float = 3.0
    model: str = "lr15"
    r0: float = 50         # km, radius at NSE exit
    T0: float = 7.0 * GK
    abar: float = None         # NSE mean nucleon number at (T0, rho0, Ye0)
    zbar: float = None         # NSE mean charge at (T0, rho0, Ye0)
    xmass_nse: dict = field(default=None, repr=False)  # {isotope: mass frac}
    t: np.ndarray = field(default=None, repr=False)
    rho: np.ndarray = field(default=None, repr=False)
    T: np.ndarray = field(default=None, repr=False)

    @classmethod
    def from_state(cls, tau, Ye0, s0, T0=7.0 * GK, **kwargs):
        """Build a Trajectory from the four physical parameters (tau, Ye0,
        s0, T0), solving rho0 <-> NSE(Abar) self-consistently and attaching
        the resulting seed composition (Abar, Zbar, full mass-fraction
        vector) from the real NSE solver (see `rho0_and_nse_state`,
        `run_nse`), instead of the crude Abar guess `rho0_from_entropy` uses
        on its own.

        Any extra dataclass fields (n, model, r0, ...) can be passed as
        keyword arguments and are forwarded unchanged.
        """
        state = rho0_and_nse_state(s0, Ye0, T0)
        return cls(Ye0=Ye0, s0=s0, tau=tau, rho0=state["rho0"], T0=T0,
                   abar=state["abar"], zbar=state["zbar"],
                   xmass_nse=state["xmass"], **kwargs)

    def build(self, t_end=5.0e-5, npts=200, t_start=1e-6):
        self.t = np.geomspace(t_start, t_end, npts)
        p = {"rho0": self.rho0, "tau": self.tau, "n": self.n}
        self.rho = DENSITY_MODELS[self.model](self.t, p)
        # T ~ rho^(1/3): adiabatic (constant-entropy) temperature history,
        # normalized so T(t_start) = T0 at rho(t_start) = rho0.
        self.T = T_from_rho(self.rho, self.rho0, self.T0)
        return self

    def kinematics(self):
        v, dlnv, t_hom = homology_diagnostic(self.t, self.rho, self.r0, self.n)
        return {"v_terminal_c": float(v[-1] / C_LIGHT), "t_hom_s": t_hom,
                "causal": bool(causality_ok(v))}
    '''
    def nu_history(self, L_nue, L_nuebar, E_nue, E_nuebar):
        r = radius_from_rho(self.t, self.rho, self.r0, self.n)
        ln, lnb = nu_capture_rates(r, L_nue, L_nuebar, E_nue, E_nuebar)
        return {"r": r, "lam_nue": ln, "lam_nuebar": lnb,
                "Ye_eq": float(ye_equilibrium(ln[0], lnb[0])),
                "exposure": float(nu_exposure(self.t, ln, lnb))}
    '''

def build_grid(box: ParameterBox, n_ye=20, n_s=20, n_tau=8, T0=7.0 * GK,
               r0=50, apply_filters=True):
    """Generate a filtered (Ye0, s0, tau, rho0) grid ready for the network."""
    ye = np.linspace(*box.ye, n_ye)
    s = np.geomspace(*box.s, n_s)
    tau = np.geomspace(*box.tau, n_tau)
    Y, S, TAU = np.meshgrid(ye, s, tau, indexing="ij")
    R0 = np.empty_like(Y)
    for idx in np.ndindex(Y.shape):
        try:
            R0[idx] = rho0_from_entropy(S[idx], Y[idx], T0)
        except ValueError:
            R0[idx] = np.nan
    mask = np.isfinite(R0)
    if apply_filters:
        mask &= admissible(Y, S, TAU, R0, r0)
    return {"Ye0": Y[mask], "s0": S[mask], "tau": TAU[mask], "rho0": R0[mask],
            "n_total": Y.size, "n_kept": int(mask.sum())}


# ============================================================================
# 10. BATCH TRAJECTORY-SET GENERATION (WinNet .dat files + index CSV)
# ============================================================================

def generate_trajectory_set(n, box: ParameterBox, outdir, component_name="",
                            T0=7.0 * GK, r0=50, model="lr15", nn=3.0,
                            t_end=5.4e-3, npts=300, t_start=2.5e-6,
                            seed=0, max_attempts=None):
    """Sample `n` physically admissible (Ye0, s0, tau) trajectories from
    `box`, solve rho0 and the seed composition self-consistently via the
    real NSE solver (`rho0_and_nse_state`) for each, build the full rho(t)
    history, and write every trajectory as a WinNet `.dat` file plus a
    summary `trajectories_index.csv` into `outdir`.

    Sampling is rejection sampling within `box` (Ye0 uniform, s0 and tau
    log-uniform): draws keep coming until `n` points survive both the NSE
    solve (rho0 root-bracketing can fail at box extremes) and
    `admissible(...)` (causality, EOS density ceiling, tau resolution
    floor), or until `max_attempts` (default 20*n) is exhausted -- so the
    function either returns exactly `n` rows or warns about the shortfall,
    it never silently returns fewer without telling you.
    """
    rng = np.random.default_rng(seed)
    max_attempts = max_attempts or 20 * n
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    attempts = 0
    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        Ye0 = rng.uniform(*box.ye)
        s0 = 10.0 ** rng.uniform(np.log10(box.s[0]), np.log10(box.s[1]))
        tau0 = 10.0 ** rng.uniform(np.log10(box.tau[0]), np.log10(box.tau[1]))
        try:
            state = rho0_and_nse_state(s0, Ye0, T0)
        except ValueError:
            continue  # rho0 not bracketed at this (s0, Ye0, T0); resample
        rho0 = state["rho0"]
        if not admissible(Ye0, s0, tau0, rho0, r0):
            continue

        i = len(rows) + 1
        tr = Trajectory(Ye0=float(Ye0), s0=float(s0), tau=float(tau0),
                        rho0=rho0, T0=T0, r0=r0, n=nn, model=model,
                        abar=state["abar"], zbar=state["zbar"],
                        xmass_nse=state["xmass"]
                        ).build(t_end=t_end, npts=npts, t_start=t_start)

        fname = out / f"traj{i:04d}.dat"
        write_winnet_trajectory(
            fname, tr.t, tr.T / GK, tr.rho, Ye=Ye0,
            r_Km=radius_from_rho(tr.t, tr.rho, tr.r0, tr.n),
            header_note=(f"{component_name} ejecta #{i}: Ye0={Ye0:.4f} "
                         f"s0={s0:.3f} kB/bar tau={tau0*1e3:.4f} ms "
                         f"rho0={rho0:.4e} g/cm3 Abar={state['abar']:.4f} "
                         f"Zbar={state['zbar']:.4f}"))
        rows.append({"id": i, "file": fname.name, "Ye0": Ye0, "s0": s0,
                    "tau_s": tau0, "rho0_gcc": rho0, "Abar": state["abar"],
                    "Zbar": state["zbar"], "ye_check": state["ye_check"],
                    "n_iter": state["n_iter"]})

    if len(rows) < n:
        warnings.warn(
            f"only generated {len(rows)}/{n} {component_name!r} trajectories "
            f"in {attempts} attempts; widen the ParameterBox or raise "
            f"max_attempts"
        )

    fieldnames = ["id", "file", "Ye0", "s0", "tau_s", "rho0_gcc", "Abar",
                 "Zbar", "ye_check", "n_iter"]
    with open(out / "trajectories_index.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} {component_name} trajectories + index to "
         f"{out}/  ({attempts} draws attempted)")
    return rows


def generate_tidal_trajectory_set(n=120, outdir=None, **kwargs):
    """Convenience wrapper: `n` tidal-ejecta trajectories, sampled from
    `ParameterBox.tidal()` (cold, neutron-rich, short-tau prior; see that
    class's docstring), written to `outdir` (default: a `Trajectories_tidal`
    folder next to this script).  See `generate_trajectory_set` for the
    rest of the options (T0, r0, model, seed, max_attempts, ...).
    """
    if outdir is None:
        outdir = Path(__file__).resolve().parent / "Trajectories_tidal"
    return generate_trajectory_set(n, ParameterBox.tidal(), outdir,
                                   component_name="tidal", **kwargs)


if __name__ == "__main__":
    # Generate 120 tidal-ejecta trajectories (ParameterBox.tidal(): cold,
    # neutron-rich, short tau), each with an NSE-consistent rho0/Abar/Zbar,
    # written as WinNet .dat files plus trajectories_index.csv.
    rows = generate_tidal_trajectory_set(n=120)
    print(f"\n  Ye0 range kept : {min(r['Ye0'] for r in rows):.4f} - {max(r['Ye0'] for r in rows):.4f}")
    print(f"  s0  range kept : {min(r['s0'] for r in rows):.3f} - {max(r['s0'] for r in rows):.3f} kB/baryon")
    print(f"  tau range kept : {min(r['tau_s'] for r in rows)*1e3:.4f} - {max(r['tau_s'] for r in rows)*1e3:.4f} ms")
    print(f"  rho0 range kept: {min(r['rho0_gcc'] for r in rows):.3e} - {max(r['rho0_gcc'] for r in rows):.3e} g/cm^3")
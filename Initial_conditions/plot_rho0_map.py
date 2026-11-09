"""
plot_rho0_map.py
================

Diagnostic heat maps of the initial state on the (s0, Ye0) plane at T0:

  Figures/rho0_map_T7GK.{png,pdf}        log10 rho0(s0, Ye0)
  Figures/nse_composition_T7GK.{png,pdf} X_n, X_alpha, X_heavy, s_ion/s0

The grid (Ye0 in [0.01, 0.5] linear, s0 in [1, 290] log) is computed once
with initial_state.InitialStateSolver and cached in rho0_grid_T7GK.npz;
re-plotting reads the cache unless --recompute is given.

Usage:  ../.venv/bin/python plot_rho0_map.py [--recompute] [--nye 49 --ns 80]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from initial_state import GK, InitialStateSolver

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "Figures"
CACHE = HERE / "rho0_grid_T7GK.npz"
T0 = 7.0 * GK
YE_RANGE, S_RANGE = (0.01, 0.50), (1.0, 290.0)
ENSEMBLE = {"Ye": (0.04, 0.42), "s": (1.0, 49.0)}

# --- reference palette (dataviz skill, references/palette.md) ---------------
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]                                    # steps 100 -> 700
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
AXIS, SURFACE = "#c3c2b7", "#fcfcfb"
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
CMAP.set_bad("#e1e0d9")                                   # failed points

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.titlesize": 10, "axes.titleweight": "bold",
})


# ============================================================================
# grid computation
# ============================================================================

def compute_grid(nye=49, ns=80):
    """Cell-centred grid whose edges span exactly YE_RANGE x S_RANGE."""
    ye_e = np.linspace(*YE_RANGE, nye + 1)
    s_e = np.geomspace(*S_RANGE, ns + 1)
    ye_c = 0.5 * (ye_e[1:] + ye_e[:-1])
    s_c = np.sqrt(s_e[1:] * s_e[:-1])
    keys = ["rho0", "Abar", "Xn", "Xp", "Xa", "Xheavy",
            "s_ion", "s_ele", "s_rad"]
    out = {k: np.full((nye, ns), np.nan) for k in keys}
    solver = InitialStateSolver(T0)
    t0, nfail = time.time(), 0
    for i, ye in enumerate(ye_c):
        curve = solver.entropy_curve(ye)
        for j, s in enumerate(s_c):
            try:
                st = solver.rho0(s, ye, curve=curve)
            except (ValueError, RuntimeError) as e:
                nfail += 1
                print(f"  fail Ye={ye:.4f} s={s:.3f}: {e}")
                continue
            for k in keys:
                out[k][i, j] = st[k]
        print(f"  Ye row {i + 1}/{nye}  ({time.time() - t0:.0f} s)", flush=True)
    np.savez(CACHE, ye_edges=ye_e, s_edges=s_e, ye=ye_c, s=s_c,
             T0=T0, **out)
    print(f"cached grid -> {CACHE}  ({nfail} failed points)")


# ============================================================================
# plotting
# ============================================================================

HALO = [pe.withStroke(linewidth=3.0, foreground=SURFACE)]


def _frame(ax, label=True):
    ax.set_xscale("log")
    ax.set_xlim(*S_RANGE)
    ax.set_ylim(*YE_RANGE)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200"])
    ax.set_xlabel(r"entropy per baryon $s_0$ [$k_B$/baryon]")
    ax.set_ylabel(r"electron fraction $Y_{e,0}$")
    ax.tick_params(which="minor", length=2)
    # ensemble sampling box
    (y0, y1), (s0, s1) = ENSEMBLE["Ye"], ENSEMBLE["s"]
    ax.add_patch(Rectangle((s0, y0), s1 - s0, y1 - y0, fill=False,
                           ec=INK, lw=1.2, ls=(0, (4, 2)), zorder=5,
                           path_effects=HALO))
    if not label:
        return
    ax.text(s1 * 1.04, y1 - 0.005, "ensemble\nrange", va="top", ha="left",
            color=INK, fontsize=7.5, zorder=6,
            bbox=dict(fc=SURFACE, ec="none", alpha=0.8, pad=1.5))


def plot_rho0(g):
    lr = np.log10(g["rho0"])
    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    vmin, vmax = np.floor(np.nanmin(lr)), np.ceil(np.nanmax(lr))
    pm = ax.pcolormesh(g["s_edges"], g["ye_edges"], lr, cmap=CMAP,
                       vmin=vmin, vmax=vmax, shading="flat", rasterized=True)
    S, Y = np.meshgrid(g["s"], g["ye"])
    lev = np.arange(vmin + 1, vmax)
    cs = ax.contour(S, Y, lr, levels=lev[lev != 12], colors=INK2,
                    linewidths=0.6, alpha=0.75)
    ax.clabel(cs, fmt=lambda v: f"$10^{{{v:.0f}}}$", fontsize=7,
              colors=INK, inline_spacing=2)
    # validity line: neutron degeneracy sets in near rho0 ~ 1e12
    c12 = ax.contour(S, Y, lr, levels=[12.0], colors=INK, linewidths=1.6)
    ax.clabel(c12, fmt={12.0: r"$10^{12}$"}, fontsize=7.5, colors=INK)
    _frame(ax)
    cb = fig.colorbar(pm, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}\,\rho_0$  [g cm$^{-3}$]", color=INK2)
    cb.outline.set_edgecolor(AXIS)
    cb.ax.tick_params(color=MUTED, labelcolor=INK2)
    fig.suptitle(r"Initial density $\rho_0(s_0, Y_{e,0})$ at "
                 rf"$T_0 = {g['T0'] / GK:.0f}$ GK", x=0.01, ha="left",
                 color=INK, fontsize=10.5, fontweight="bold")
    ax.set_title(r"Bold line: $\rho_0 = 10^{12}$ g cm$^{-3}$, onset of "
                 r"neutron degeneracy (Maxwell-Boltzmann ion entropy "
                 "unreliable to its left)", loc="left", fontsize=7.5,
                 color=INK2, fontweight="normal")
    _save(fig, "rho0_map_T7GK")


def plot_composition(g):
    panels = [("Xn", r"free neutrons $X_n$"),
              ("Xa", r"alphas $X_\alpha$"),
              ("Xheavy", r"heavy nuclei $X_{Z>2}$"),
              ("s_ion_frac", r"ion share of entropy $s_\mathrm{ion}/s_0$")]
    g = dict(g)
    g["s_ion_frac"] = g["s_ion"] / (g["s_ion"] + g["s_ele"] + g["s_rad"])
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), sharex=True,
                             sharey=True, constrained_layout=True)
    for ax, (k, title) in zip(axes.flat, panels):
        pm = ax.pcolormesh(g["s_edges"], g["ye_edges"], g[k], cmap=CMAP,
                           vmin=0, vmax=1, shading="flat", rasterized=True)
        _frame(ax, label=(k == "Xn"))
        ax.set_title(title, loc="left")
        cb = fig.colorbar(pm, ax=ax, pad=0.02, ticks=[0, 0.25, 0.5, 0.75, 1])
        cb.outline.set_edgecolor(AXIS)
        cb.ax.tick_params(color=MUTED, labelcolor=INK2)
    for ax in axes[0]:
        ax.set_xlabel("")
    for ax in axes[:, 1]:
        ax.set_ylabel("")
    fig.suptitle(rf"NSE state at freeze-out, $T_0 = {g['T0'] / GK:.0f}$ GK "
                 r"(mass fractions and ion entropy share)",
                 x=0.01, ha="left", color=INK, fontsize=10.5,
                 fontweight="bold")
    _save(fig, "nse_composition_T7GK")


def _save(fig, stem):
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {FIG / stem}.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--nye", type=int, default=49)
    ap.add_argument("--ns", type=int, default=80)
    a = ap.parse_args()
    if a.recompute or not CACHE.exists():
        compute_grid(a.nye, a.ns)
    g = dict(np.load(CACHE))
    plot_rho0(g)
    plot_composition(g)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Publication figure: linear response and transport across three systems.

Rows are diamond, silicon and aluminium on cells of about a thousand atoms
(1,024 / 1,024 / 729), the same operators the spectrum figure uses.
Columns: the Kubo-Greenwood conductivity sigma(E) with the finite-T dc
point riding it at E_F; the optical conductivity sigma(omega), real and
imaginary parts in one panel; the ballistic diffusion D(E, t) at three
energies. Every curve is the 64-sample stochastic estimator with the
commutator velocity at Gamma. Two references sit under each: the pale grey
band is the trace limit from dense diagonalization of the same supercell
(make_ref_curves.py, sampling error only), the pale warm band the bulk
limit from the primitive cell on a dense k-mesh with matched broadening
(make_bulk_ref.py, finite-cell and kernel error included).
Same design language as fig_ldos_gallery: single accent, 180 mm double
column, Arial 5-6 pt. Deviations are quoted in README.md, not on the
panels.
"""
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.join(HERE, "results", *p)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

INK, INK2, MUTED, FRAME = "#1a1a1a", "#4d4c49", "#8a8880", "#c9c8bf"
BLUE, ACCENT, REFPALE = "#2a78d6", "#d55e00", "#b9b7ae"
BULK = "#e8c9a0"                                   # k-mesh bulk reference (warm, pale)

# Convention. Every transport quantity is the spin-degenerate SI value (the
# code carries the pi of Kubo-Greenwood and the spin factor 2); nothing is
# rescaled here.
SCALE_SIGMA_E = 1.0
SCALE_SIGMA_W = 1.0
IMAG = "#1baf7a"                                   # Im sigma(omega)
TIME_BLUES = ["#9ec5f4", "#5598e7", "#2a78d6", "#104281"]
CM1_TO_EV = 1.2398419843e-4

SYSTEMS = [
    dict(letter="a", name="diamond", sub="1,024 atoms · insulator",
         run="diamond", mu=15.418, d_energies=[-4.0, 2.5, 5.0]),
    dict(letter="b", name="silicon", sub="1,024 atoms · semiconductor",
         run="silicon", mu=6.528, d_energies=[-2.0, 1.0, 4.5]),
    dict(letter="c", name="aluminum", sub="729 atoms · metal",
         run="aluminum", mu=11.112081, d_energies=[-4.0, 0.0, 4.0]),
]
EWIN = 8.0    # eV around E_F for sigma(E) and D(E)
WWIN = 12.0   # eV photon-energy window for the optical panel

# D(E, t) is a ratio of spectral densities and is noise where the DOS
# vanishes. The mask comes from the exact eigenvalues of the same cell
# (ref_*_eig.npz), broadened by 0.25 eV: keep energies above 5% of the
# median in-window DOS.
DOS_SMOOTH_EV = 0.25
DOS_MASK_FRAC = 0.05


def dos_mask(run, energies_abs, mu):
    eps = np.load(R(f"ref_{run}_eig.npz"))["eps"]
    e = np.linspace(eps.min() - 1.0, eps.max() + 1.0, 20000)
    d = np.histogram(eps, bins=e.size, range=(e[0], e[-1]))[0].astype(float)
    ds = gaussian_filter1d(d, DOS_SMOOTH_EV / (e[1] - e[0]))
    win = np.abs(e - mu) < EWIN
    thr = DOS_MASK_FRAC * np.median(ds[win])
    return np.interp(energies_abs, e, ds, left=0.0, right=0.0) >= thr


def voigt_diag_mean(s6):
    return (s6[..., 0] + s6[..., 3] + s6[..., 5]) / 3.0


def style(ax, last_row):
    ax.tick_params(labelsize=5.0, color=FRAME, labelcolor=MUTED, length=1.8,
                   pad=1.5)
    for sp in ax.spines.values():
        sp.set_color(FRAME)
    if not last_row:
        ax.tick_params(labelbottom=False)
    ax.grid(color="#eceae4", lw=0.4)
    ax.set_axisbelow(True)


def chip(ax, text, x=0.04, y=0.955, size=5.5, color=INK, ha="left",
         weight="normal"):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, color=color,
            va="top", ha=ha, fontweight=weight,
            bbox=dict(facecolor="white", alpha=0.82, lw=0, pad=1.3))


def main():
    rows = [s for s in SYSTEMS if os.path.exists(R(s["run"], "ac.npz"))]
    n_r = len(rows)
    fig_w, fig_h = 7.05, 1.15 * n_r + 0.55
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    gs = fig.add_gridspec(n_r, 3, wspace=0.28, hspace=0.10,
                          left=0.075, right=0.985, top=0.965,
                          bottom=0.32 / fig_h)
    last = n_r - 1

    for r, sysd in enumerate(rows):
        mu = sysd["mu"]
        zs = np.load(R(sysd["run"], "dc_spectrum.npz"))
        e = zs["energy"] - mu
        sig_e = voigt_diag_mean(zs["sigma"][0])
        dc = voigt_diag_mean(np.load(R(sysd["run"], "dc.npz"))["sigma"][0])
        za = np.load(R(sysd["run"], "ac.npz"))
        w = za["omega_cm1"] * CM1_TO_EV
        sig_w = voigt_diag_mean(za["sigma"][0])
        zm = np.load(R(sysd["run"], "msd.npz"))
        em = zm["energy"] - mu
        t = zm["time_fs"]
        diff = zm["diffusion"][0]  # (t, E)
        p_ref = R(f"ref_{sysd['run']}_curves.npz")
        ref = np.load(p_ref) if os.path.exists(p_ref) else None
        p_bulk = R(f"bulk_{sysd['run']}_curves.npz")
        bulk = np.load(p_bulk) if os.path.exists(p_bulk) else None
        sig_e = sig_e * SCALE_SIGMA_E
        dc = dc * SCALE_SIGMA_E
        sig_w = sig_w * SCALE_SIGMA_W

        # -- col 0: Kubo-Greenwood sigma(E), dc point cross-check ----------
        ax = fig.add_subplot(gs[r, 0])
        win = np.abs(e) < EWIN
        if bulk is not None:
            ax.plot(e[win], bulk["sigma_e"][win] / 1e6, color=BULK, lw=1.5,
                    solid_capstyle="round", zorder=0)
        if ref is not None:
            ax.plot(e[win], SCALE_SIGMA_E * ref["sigma_e"][win] / 1e6, color=REFPALE,
                    lw=1.5, solid_capstyle="round", zorder=1)
        ax.plot(e[win], sig_e[win] / 1e6, color=BLUE, lw=0.8, zorder=2)
        ax.axvline(0.0, color=INK2, lw=0.6, ls=(0, (4, 2.4)))
        ax.plot([0.0], [dc / 1e6], marker="o", ms=3.0, color=ACCENT,
                mec="white", mew=0.4, zorder=5, ls="none")
        ax.set_xlim(-EWIN, EWIN)
        ax.set_ylim(bottom=min(0.0, 1.1 * sig_e[win].min() / 1e6))
        style(ax, r == last)
        chip(ax, f"{sysd['letter']}", size=8.0, weight="bold")
        ax.text(0.13, 0.955, f"{sysd['name']}\n{sysd['sub']}",
                transform=ax.transAxes, fontsize=5.4, color=INK2, va="top",
                linespacing=1.25,
                bbox=dict(facecolor="white", alpha=0.82, lw=0, pad=1.3))
        ax.set_ylabel("$\\sigma(E)$ (MS m$^{-1}$)", fontsize=6.0, color=INK2,
                      labelpad=2.0)
        if r == 0:
            chip(ax, "$\\sigma_{dc}(\\mu, 300\\,$K$)$", x=0.97, y=0.955,
                 size=5.2, color=ACCENT, ha="right")
        if r == last:
            ax.set_xlabel("$E-E_F$ (eV)", fontsize=6.0, color=INK2)

        # -- col 1: optical conductivity, real and imaginary parts ---------
        ax = fig.add_subplot(gs[r, 1])
        wwin = w <= WWIN
        if bulk is not None:
            # the bulk optical curves on their own fine grid: the Drude line
            # of a metal is narrower than the run's omega step
            fine = "omega_fine" in bulk.files
            wf = bulk["omega_fine"] if fine else w
            rf = bulk["sigma_w_fine"] if fine else bulk["sigma_w"]
            imf = bulk["sigma_w_imag_fine"] if fine else bulk["sigma_w_imag"]
            wf_win = wf <= WWIN
            ax.plot(wf[wf_win], rf[wf_win] / 1e6, color=BULK, lw=1.5,
                    solid_capstyle="round", zorder=0)
            ax.plot(wf[wf_win], imf[wf_win] / 1e6, color=BULK,
                    lw=1.5, solid_capstyle="round", zorder=0)
        if ref is not None:
            ax.plot(w[wwin], SCALE_SIGMA_W * ref["sigma_w"][wwin] / 1e6,
                    color=REFPALE, lw=1.5, solid_capstyle="round", zorder=1)
            if "sigma_w_imag" in ref.files:
                ax.plot(w[wwin], SCALE_SIGMA_W * ref["sigma_w_imag"][wwin] / 1e6,
                        color=REFPALE, lw=1.5, solid_capstyle="round",
                        zorder=1)
        ax.plot(w[wwin], sig_w[wwin].real / 1e6, color=BLUE, lw=0.8,
                zorder=2)
        ax.plot(w[wwin], sig_w[wwin].imag / 1e6, color=IMAG, lw=0.8,
                zorder=2)
        ax.axhline(0.0, color=FRAME, lw=0.5, zorder=0)
        ax.set_xlim(0, WWIN)
        # The y-range follows the run and the bulk interband response; a
        # metal's resolved Drude line (bulk, omega < 0.5 eV) would otherwise
        # set the scale for the whole panel, so it runs off the top and its
        # height is written in.
        top = 1.15 * max(sig_w[wwin].real.max(), sig_w[wwin].imag.max()) / 1e6
        if bulk is not None:
            inter = wf > 0.5
            top = max(top, 1.15 * max(rf[wf_win & inter].max(), imf[wf_win & inter].max()) / 1e6)
            # The bulk curve is the regular part; a metal's Drude weight D of
            # the D delta(omega) term (which the finite-time transform of the
            # run cannot represent either) is written in.
            if "drude_weight" in bulk.files and float(bulk["drude_weight"]) > 1e-3 * 1e6:
                ax.text(0.30, 0.955, f"$D$ = {float(bulk['drude_weight']) / 1e6:.2f} MS m$^{{-1}}$ eV",
                        transform=ax.transAxes, fontsize=5.0, color=MUTED, va="top")
        bot = min(sig_w[wwin].imag.min(), imf[wf_win].min() if bulk is not None else 0.0) / 1e6
        ax.set_ylim(1.15 * bot, top)
        style(ax, r == last)
        handles = [Line2D([], [], color=BLUE, lw=0.8, label="Re $\\sigma$"),
                   Line2D([], [], color=IMAG, lw=0.8, label="Im $\\sigma$")]
        if r == 0:
            handles.insert(0, Line2D([], [], color=BULK, lw=1.5,
                                     label="bulk, $k$-mesh"))
            handles.insert(0, Line2D([], [], color=REFPALE, lw=1.5,
                                     label="same cell, exact"))
        ax.legend(handles=handles, fontsize=5.0, frameon=False,
                  loc="upper left", handlelength=1.4, borderpad=0.1,
                  labelspacing=0.3)
        ax.set_ylabel("$\\sigma(\\omega)$ (MS m$^{-1}$)", fontsize=6.0,
                      color=INK2, labelpad=2.0)
        if r == last:
            ax.set_xlabel("$\\hbar\\omega$ (eV)", fontsize=6.0, color=INK2)

        # -- col 2: ballistic diffusion D(t) at selected energies ----------
        # Requested energies snap to the nearest grid point that has
        # states (DOS mask) and strictly positive D over the whole run.
        ax = fig.add_subplot(gs[r, 2])
        keep = dos_mask(sysd["run"], zm["energy"], mu)
        dmax = 0.0
        for ec, ev in zip(TIME_BLUES[1:], sysd["d_energies"]):
            cand = np.flatnonzero(keep & (diff[1:] > 0).all(axis=0))
            i_e = cand[np.argmin(np.abs(em[cand] - ev))]
            if bulk is not None:
                ax.plot(t[1:], bulk["diffusion"][1:, i_e], color=BULK,
                        lw=1.5, solid_capstyle="round", zorder=0)
            if ref is not None:
                ax.plot(t[1:], ref["diffusion"][1:, i_e], color=REFPALE,
                        lw=1.5, solid_capstyle="round", zorder=1)
            ax.plot(t[1:], diff[1:, i_e], color=ec, lw=0.8, zorder=2,
                    label=f"{em[i_e]:+.1f} eV")
            dmax = max(dmax, diff[1:, i_e].max())
        ax.set_xlim(0, t[-1])
        ax.set_ylim(0.0, 1.06 * dmax)
        style(ax, r == last)
        ax.set_ylabel("$D(E,t)$ (Å$^2$ fs$^{-1}$)", fontsize=6.0, color=INK2,
                      labelpad=2.0)
        ax.legend(fontsize=5.0, frameon=False, loc="upper left",
                  handlelength=1.2, borderpad=0.1, labelspacing=0.25,
                  title="$E-E_F$" if r == 0 else None, title_fontsize=5.0)
        if r == last:
            ax.set_xlabel("$t$ (fs)", fontsize=6.0, color=INK2)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_transport.{ext}"))
    print(f"wrote fig_transport ({n_r} rows)")


if __name__ == "__main__":
    main()

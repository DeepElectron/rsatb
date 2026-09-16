#!/usr/bin/env python3
"""Publication figure: spectrum + charge density + LDOS across three systems.

Rows are systems (diamond insulator, silicon semiconductor, aluminum
metal); columns are [DOS | PDOS | charge density | two LDOS maps |
line cut]. The DOS and PDOS are the STOCHASTIC estimator's own curves
(pdos task, Lowdin populations) validated against dense diagonalization
rebuilt through the SAME Jackson-KPM kernel (kpm_ref) -- what the eye
compares is the estimator's error and nothing else. Both routes get the
same extra 0.2 eV Gaussian presentation broadening, which cannot
flatter either one. The map panels are the PRODUCTION real-space
estimator, each annotated with its pointwise relative deviation from
the diagonalization reference.

Layout: every periodic slab is resampled onto its rectangular display
cell (the display domain of a periodic map is free; the rectangle tiles
the plane exactly), so panels have NO shear triangles and -- all three
lattices being fcc-sheared with a2_perp/|a1| = sin 60 -- every map
shares one aspect. Sized for a 180 mm double column (Nature).

Encoding: map magnitude is a single-hue sequential blue (light = zero);
the diagonalization reference is always PALE and THICK, this work
always the thin saturated line on top -- route identity by weight, so
hue is free to carry shell identity in the PDOS (validated categorical
s/p/d). The one accent (vermillion) marks probed window energies in the
DOS. Panel letters a/b/c per row follow journal convention.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter1d

from kpm_ref import kpm_curves, n_moments_for, load_ranks

HERE = os.path.dirname(os.path.abspath(__file__))
def R(*p):
    """Result path: the runs and the diagonalization references (written by
    make_references.sh) both live under results/."""
    return os.path.join(HERE, "results", *p)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

INK, INK2, MUTED, FRAME = "#1a1a1a", "#4d4c49", "#8a8880", "#c9c8bf"
# LDOS maps -- the method's result -- carry the saturated focus blue; the
# charge density gets a calmer sage-green ramp so the two map kinds read apart
# without adding a loud new hue (green reuses the d-shell family).
SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
    "#0d366b"])
SEQ_DEN = LinearSegmentedColormap.from_list("seq_sage", [
    "#eef4ef", "#dcebe0", "#c6ddcb", "#aecdb4", "#94bd9d", "#79ac86",
    "#5f9c70", "#4c9a6a", "#3d8459", "#2f6e49", "#245a3b", "#1a472e",
    "#123821"])
BLUE, REFPALE, ACCENT = "#2a78d6", "#b9b7ae", "#b0413f"
DOSFILL = "#e9e8e1"
# Muted editorial trio (blue kept as the "this work" identity; p/d softened
# from the earlier neon orange/green to the seaborn-deep amber and sage).
SHELL_COLOR = {"s": "#2a78d6", "p": "#dd8452", "d": "#4c9a6a"}

DE_TASK = 0.1  # the pdos runs' own resolution
FW = 0.2       # presentation broadening, both routes identically

# Orbital-slot -> shell maps, from the basis files' radial ordering.
SHELLS_2S2P1D = {"s": [0, 1], "p": [2, 3, 4, 5, 6, 7],
                 "d": [8, 9, 10, 11, 12]}
SHELLS_4S4P1D = {"s": [0, 1, 2, 3], "p": list(range(4, 16)),
                 "d": list(range(16, 21))}

SYSTEMS = [
    dict(letter="a", name="diamond, 3,456 atoms", kind="insulator",
         ref="dia_rs_k12", run="dia_ldos", mu=13.377, gap_center=True,
         den_n_atom=3456,
         pdos_run="dia_pdos", pdos_de=0.3, pdos_n_basis=44928,
         pdos_n_atom=3456, pdos_ref=("dia_rs_k12", 44928, 3456),
         shells=SHELLS_2S2P1D, n_basis=13312, n_atom=1024,
         show=[0, 1],
         den_ref=("dia_rs_k12", "k_density.f64", "k_grid.json"),
         den_run="dia_density",
         a1=np.array([2.52685, 0.0, 0.0]) * 12,
         a2=np.array([1.2634250, 2.1883160, 0.0]) * 12),
    dict(letter="b", name="silicon, 3,456 atoms", kind="semiconductor",
         ref="si_rs_k12", run="si_ldos", mu=6.51913, gap_center=True,
         den_n_atom=3456,
         pdos_run="si_pdos", pdos_de=0.3, pdos_n_basis=44928,
         pdos_n_atom=3456, pdos_ref=("si_rs_k12", 44928, 3456),
         shells=SHELLS_2S2P1D, n_basis=1664, n_atom=128,
         show=[0, 1],
         den_ref=("si_rs_k12", "k_density.f64", "k_grid.json"),
         den_run="si_density",
         a1=np.array([3.8669493198, 0.0, 0.0]) * 12,
         a2=np.array([1.9334796667, 3.3488819599, 0.0]) * 12),
    dict(letter="c", name="aluminum, 1,728 atoms", kind="metal",
         ref="al_rs_k12", run="al_ldos", mu=11.112081,
         pdos_run="al_pdos", pdos_de=0.3, pdos_n_basis=36288,
         pdos_n_atom=1728, pdos_ref=("al_rs_k12", 36288, 1728),
         shells=SHELLS_4S4P1D, n_basis=36288, n_atom=1728,
         show=[0, 2],  # Fermi level and the +4 eV window
         den_ref=("al_rs_k12", "k_density.f64", "k_grid.json"),
         den_run="al_density",
         a1=np.array([2.85595, 0.0, 0.0]) * 12,
         a2=np.array([1.42798, 2.47333, 0.0]) * 12,
         zoom=0.5),
]

ASPECT = np.sin(np.deg2rad(60.0))  # a2_perp/|a1|, identical for all rows


def to_rect(plane, a1, a2, zoom=1.0, nu=240, nv=208):
    """Resample a periodic sheared slab onto its rectangular display cell.

    The map is periodic in both lattice vectors, so the display domain is
    free; the rectangle u in [0,|a1|) x v in [0,a2_perp) tiles the plane
    exactly and wastes no panel area on shear triangles. Periodic
    bilinear interpolation of the fractional-grid data. The display
    WINDOW zooms; the inverse map to fractional coordinates keeps the
    FULL cell scales, or the window would relabel (and shear) the whole
    cell instead of magnifying it.
    """
    n0, n1 = plane.shape
    L1 = np.linalg.norm(a1)
    e1 = a1 / L1
    a2x = a2 @ e1
    a2y = np.linalg.norm(a2 - a2x * e1)
    Lz, az = zoom * L1, zoom * a2y
    u = (np.arange(nu) + 0.5) / nu * Lz
    v = (np.arange(nv) + 0.5) / nv * az
    Vg, Ug = np.meshgrid(v, u)
    f1 = Vg / a2y
    f0 = (Ug - f1 * a2x) / L1
    g0, g1 = f0 * n0, f1 * n1
    i0 = np.floor(g0).astype(int)
    t0 = g0 - i0
    i1 = np.floor(g1).astype(int)
    t1 = g1 - i1
    i0 %= n0
    i1 %= n1
    j0, j1 = (i0 + 1) % n0, (i1 + 1) % n1
    out = ((1 - t0) * (1 - t1) * plane[i0, i1] +
           t0 * (1 - t1) * plane[j0, i1] +
           (1 - t0) * t1 * plane[i0, j1] +
           t0 * t1 * plane[j0, j1])
    return out.T, Lz, az


def style_axes(ax, r, c):
    ax.tick_params(labelsize=5.0, color=FRAME, labelcolor=MUTED, length=1.8,
                   pad=1.5, direction="in")
    for sp in ax.spines.values():
        sp.set_color(FRAME)
    if r != 2:
        ax.tick_params(labelbottom=False)
    if c != 3:
        ax.tick_params(labelleft=False)


def in_panel_label(ax, text, loc="top", color=INK, size=5.5):
    y, va = (0.965, "top") if loc == "top" else (0.045, "bottom")
    ax.text(0.04, y, text, transform=ax.transAxes, fontsize=size, color=color,
            va=va, bbox=dict(facecolor="white", alpha=0.82, lw=0, pad=1.3))


def draw_rect_map(ax, plane, a1, a2, vmax, r, c, zoom=1.0, cmap=SEQ):
    img, L1, a2y = to_rect(plane, a1, a2, zoom)
    ax.imshow(img, origin="lower", extent=[0, L1, 0, a2y], cmap=cmap,
              vmin=0.0, vmax=max(vmax, 1e-30), aspect="auto",
              interpolation="bilinear", rasterized=True)
    style_axes(ax, r, c)


def spectrum_curves(sysd, mu):
    """This work's DOS + s/p/d PDOS (pdos task, Lowdin) and the reference
    rebuilt through the SAME Jackson kernel, per atom per eV; smoothed by
    FW identically. Returns (E - E_F, ours_tot, ours_shell, ref_tot,
    ref_shell, dev_rms_of_peak[unsmoothed])."""
    z = np.load(R(sysd["pdos_run"], "pdos_slot.npz"))
    e = z["energy"]
    de = e[1] - e[0]
    scale = e[e.size // 2] - e[0]
    n_basis = sysd.get("pdos_n_basis", sysd["n_basis"])
    n_atom = sysd.get("pdos_n_atom", sysd["n_atom"])
    de_task = sysd.get("pdos_de", DE_TASK)
    to_atom = n_basis / (2.0 * scale) / n_atom
    ours_tot = z["total"][0] * to_atom
    ours_shell = np.stack([z["pdos"][0][s].sum(0)
                           for s in sysd["shells"].values()]) * to_atom

    # The reference is the exact spectrum of the SAME cell the run used --
    # the primitive pencil on the commensurate n^3 mesh, which is the same
    # eigenproblem by band folding. Both sides then share the moment count,
    # the Jackson kernel, the edge inset and the 2*rho convention, and differ
    # ONLY in how the Chebyshev moments were obtained: stochastically traced
    # here, summed exactly over eigenvalues there. So "dev" is the estimator
    # error alone. A denser reference would be a different physical system
    # (closer to the infinite-crystal limit), and folding that distance into
    # the same number invites the reader to charge the cell's finite size to
    # the method: measured on this figure, estimator 0.2% against cell-to-
    # converged 5-9% (compare_spectrum.py prints both).
    ref_dir, ref_nb, ref_na = sysd["pdos_ref"]
    cache = R(f"ref_curves_{sysd['pdos_run']}.npz")
    if os.path.exists(cache):
        zc = np.load(cache)
        ok = zc["energy"].shape == e.shape and np.allclose(zc["energy"], e)
    else:
        ok = False
    if ok:
        ref_tot, ref_shell = zc["total"], zc["shell"]
    else:
        kdir = R(ref_dir)
        eps = np.fromfile(os.path.join(kdir, "k_eigenvalues.f64"))
        n_slot = len(sum(sysd["shells"].values(), []))
        w = np.fromfile(os.path.join(kdir, "k_lowdin_weight_slot.f64"))
        w = w.reshape(n_slot, -1)  # [slot, state], as plot_validation.py
        rows = np.vstack([np.ones((1, len(eps)))] +
                         [w[s].sum(0)[None, :]
                          for s in sysd["shells"].values()])
        n_mom = n_moments_for(e, de_task)
        ref_to_atom = ref_nb / (2.0 * scale) / ref_na
        out = kpm_curves(eps, rows, e, n_mom, ref_nb) * ref_to_atom
        ref_tot, ref_shell = out[0], out[1:]
        np.savez(cache, energy=e, total=ref_tot, shell=ref_shell)

    dev = np.sqrt(((ours_tot - ref_tot) ** 2).mean()) / ref_tot.max()
    sm = lambda y: gaussian_filter1d(y, (FW / 2.3548) / de, mode="nearest")
    return (e - mu, sm(ours_tot),
            np.stack([sm(y) for y in ours_shell]), sm(ref_tot),
            np.stack([sm(y) for y in ref_shell]), dev)


def style_spectrum(ax, r, e_lo, e_hi, labelleft):
    ax.set_ylim(e_lo, e_hi)
    ax.invert_xaxis()  # curves grow toward the maps they feed
    ax.tick_params(labelsize=5.0, color=FRAME, labelcolor=MUTED,
                   length=1.8, pad=1.5, labelbottom=False, bottom=False,
                   labelleft=labelleft, direction="in")
    for sp in ax.spines.values():
        sp.set_color(FRAME)
    ax.axhline(0.0, color=INK2, lw=0.6, ls=(0, (4, 2.4)))


def main():
    # Sized for a 180 mm double column: 7 panels per row plus a slim spacer
    # between the spectrum pair and the maps; maps ASPECT-high.
    widths = [0.72, 0.72, 0.30, 1.0, 1.0, 1.0, 1.18]  # col 2 = spacer
    fig_w, fig_h = 7.05, 3.16
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    gs = fig.add_gridspec(3, 7, width_ratios=widths, wspace=0.06,
                          hspace=0.08, left=0.055, right=0.925, top=0.955,
                          bottom=0.095)

    for r, sysd in enumerate(SYSTEMS):
        grid = json.load(open(R(sysd["ref"], "k_grid.json")))
        shape = tuple(grid["shape"])
        npt = int(np.prod(shape))
        ref = np.fromfile(R(sysd["ref"], "k_ldos.f64")).reshape(-1, npt)
        en = np.fromfile(R(sysd["ref"], "k_ldos_energies.f64"))
        ours = load_ranks(R(sysd["run"]), "ldos.rank*.npz", "ldos", npt)
        # Gapped systems: E_F displayed at the exact gap CENTER, derived
        # from the reference eigenvalues (midpoint of the levels enclosing
        # the filling that the seed mu selects) -- self-maintaining, and mu
        # only ever affects labels and axes, never any computed map.
        mu = sysd["mu"]
        if sysd.get("gap_center"):
            eig = np.sort(np.fromfile(R(sysd["ref"], "k_eigenvalues.f64")))
            nb = int((eig < mu).sum())
            mu = 0.5 * (eig[nb - 1] + eig[nb])
        probes = en - mu
        order = np.argsort(np.abs(probes))
        shown = probes[[order[i] for i in sysd["show"]]]
        e_lo = min(shown.min(), 0.0) - 1.8
        e_hi = max(shown.max(), 0.0) + 1.8

        x, o_tot, o_sh, r_tot, r_sh, dos_dev = spectrum_curves(sysd, mu)
        win = (x > e_lo - 0.5) & (x < e_hi + 0.5)

        # Column 0: total DOS, both routes. The window energies of the LDOS
        # maps are flagged with a triangle at the zero-DOS (right) spine -- a
        # marker, not a segment, so the eye never confuses it with a curve.
        ax = fig.add_subplot(gs[r, 0])
        ax.fill_betweenx(x[win], 0.0, r_tot[win], color=DOSFILL, lw=0)
        ax.plot(r_tot[win], x[win], color=REFPALE, lw=1.8, alpha=0.9,
                solid_capstyle="round")
        ax.plot(o_tot[win], x[win], color=BLUE, lw=0.7)
        dmax = r_tot[win].max()
        ax.set_xlim(0.0, 1.12 * dmax)
        for pk in shown:
            ax.plot(0.045 * dmax, pk, marker="<", ms=4.5, color=ACCENT,
                    mec="white", mew=0.5, clip_on=False, zorder=5)
        style_spectrum(ax, r, e_lo, e_hi, labelleft=True)
        ax.set_ylabel("$E-E_F$ (eV)", fontsize=6.0, color=INK2, labelpad=1.5)
        if r == 2:
            ax.set_xlabel("DOS", fontsize=6.0, color=INK2, labelpad=2.0)
        # Identity sits in the empty right margin (DOS -> 0 at the right spine
        # after inversion); letter top-left, deviation bottom-right -- none of
        # the three crosses the curve. Two short lines replace the old block.
        BOX = dict(facecolor="white", alpha=0.9, lw=0, pad=1.2)
        ax.text(0.07, 0.965, sysd["letter"], transform=ax.transAxes,
                fontsize=8.0, fontweight="bold", color=INK, va="top", bbox=BOX)
        # Three evenly spaced lines in the right margin: name, kind, size.
        ax.text(0.95, 0.955, sysd["name"].split(",")[0], transform=ax.transAxes,
                fontsize=6.3, color=INK, va="top", ha="right",
                fontweight="bold", bbox=BOX)
        ax.text(0.95, 0.840, sysd["kind"], transform=ax.transAxes,
                fontsize=5.2, color=INK2, va="top", ha="right", bbox=BOX)
        ax.text(0.95, 0.740, f"{sysd['pdos_n_atom']:,} atoms",
                transform=ax.transAxes, fontsize=5.2, color=INK2, va="top",
                ha="right", bbox=BOX)
        ax.text(0.95, 0.04, f"dev {100*dos_dev:.1f}%", transform=ax.transAxes,
                fontsize=5.0, color=INK, va="bottom", ha="right", bbox=BOX)

        # Column 1: s/p/d PDOS, both routes, shell = hue, route = weight.
        ax = fig.add_subplot(gs[r, 1])
        pmax = r_sh[:, win].max()
        for (nm, _), rc, oc in zip(sysd["shells"].items(), r_sh, o_sh):
            ax.plot(rc[win], x[win], color=SHELL_COLOR[nm], lw=1.8,
                    alpha=0.32, solid_capstyle="round")
            ax.plot(oc[win], x[win], color=SHELL_COLOR[nm], lw=0.7)
        for i2, nm in enumerate(sysd["shells"]):
            ax.text(0.10 + 0.17 * i2, 0.965, nm, transform=ax.transAxes,
                    fontsize=6.0, color=SHELL_COLOR[nm], weight="bold",
                    va="top",
                    bbox=dict(facecolor="white", alpha=0.82, lw=0, pad=1.2))
        ax.set_xlim(0.0, 1.14 * pmax)
        style_spectrum(ax, r, e_lo, e_hi, labelleft=False)
        if r == 2:
            ax.set_xlabel("PDOS", fontsize=6.0, color=INK2, labelpad=2.0)

        # Column 3: charge density from the same machinery.
        dref_dir, dref_file, dref_grid = sysd["den_ref"]
        dshape = tuple(json.load(open(R(dref_dir, dref_grid)))["shape"])
        dref = np.fromfile(R(dref_dir, dref_file)).reshape(dshape)
        dours = load_ranks(R(sysd["den_run"]), "density.rank*.npz", "density",
                           int(np.prod(dshape))).reshape(dshape)
        izd = int(np.argmax(dref.sum(axis=(0, 1))))
        ax = fig.add_subplot(gs[r, 3])
        draw_rect_map(ax, dours[:, :, izd], sysd.get("den_a1", sysd["a1"]),
                      sysd.get("den_a2", sysd["a2"]),
                      np.percentile(dours[:, :, izd], 99.5), r, 3,
                      sysd.get("zoom", 1.0), cmap=SEQ_DEN)
        # Atom count lives once, in the DOS panel; not repeated here.
        in_panel_label(ax, "density $n(\\mathbf{r})$")
        dmask = dref > 0.2 * dref.max()
        ddev = 100.0 * ((dours - dref)[dmask] / dref[dmask]).std()
        in_panel_label(ax, f"dev {ddev:.2f}%", loc="bottom")
        ax.set_ylabel("v (Å)", fontsize=6.0, color=INK2, labelpad=1.5)
        if r == 2:
            ax.set_xlabel("u (Å)", fontsize=6.0, color=INK2)

        # Columns 4-5: two representative LDOS windows (Fermi-nearest
        # ordering; sysd["show"] picks which, skipping e.g. empty windows).
        cut_k = order[0]
        row_vmax = None
        for c, oi in enumerate(sysd["show"]):
            k = order[oi]
            b = ours[k].reshape(shape)
            a = ref[k].reshape(shape)
            iz = int(np.argmax(a.sum(axis=(0, 1))))
            vmax = np.percentile(b[:, :, iz], 99.5)
            if row_vmax is None:
                row_vmax = vmax
            ax = fig.add_subplot(gs[r, c + 4])
            draw_rect_map(ax, b[:, :, iz], sysd["a1"], sysd["a2"], vmax, r,
                          c + 4, sysd.get("zoom", 1.0))
            in_panel_label(ax, f"$E-E_F$ = {probes[k]:+.2f} eV")
            mask = ref[k] > 0.2 * ref[k].max()
            dev = 100.0 * ((ours[k] - ref[k])[mask] / ref[k][mask]).std()
            note = f"dev {dev:.2f}%" if dev >= 0.005 else "dev < 0.005%"
            in_panel_label(ax, note, loc="bottom")
            if r == 2:
                ax.set_xlabel("u (Å)", fontsize=6.0, color=INK2)

        # Column 6: line cut through the reference maximum.
        a = ref[cut_k].reshape(shape)
        b = ours[cut_k].reshape(shape)
        _, i1m, izm = np.unravel_index(np.argmax(a), shape)
        xg = np.arange(shape[0]) / shape[0] * np.linalg.norm(sysd["a1"])
        ax = fig.add_subplot(gs[r, 6])
        ax.plot(xg, a[:, i1m, izm], color=REFPALE, lw=2.6,
                solid_capstyle="round", zorder=1)
        ax.plot(xg, b[:, i1m, izm], color=BLUE, lw=1.0, zorder=2)
        zoom = sysd.get("zoom", 1.0)
        ax.set_xlim(0.0, zoom * np.linalg.norm(sysd["a1"]))
        ax.set_ylim(0, 1.24 * a[:, i1m, izm].max())
        ax.grid(color="#eceae4", lw=0.4)
        ax.set_axisbelow(True)
        style_axes(ax, r, 6)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.tick_params(labelright=True, labelsize=5.0, labelcolor=MUTED,
                       direction="in")
        in_panel_label(ax, f"$E-E_F$ = {probes[cut_k]:+.2f} eV")
        if r == 0:
            ax.legend(handles=[
                Line2D([], [], color=REFPALE, lw=2.6, label="diagonalization"),
                Line2D([], [], color=BLUE, lw=1.0, label="this work"),
            ], fontsize=5.5, frameon=False, ncol=2, loc="lower center",
                bbox_to_anchor=(0.5, 1.02), handlelength=1.4,
                columnspacing=0.8, borderpad=0.1)
        ax.set_ylabel("LDOS (eV$^{-1}$ Å$^{-3}$)", fontsize=5.8,
                      color=INK2, labelpad=2.0)
        if r == 2:
            ax.set_xlabel("distance along $\\mathbf{a}_1$ (Å)",
                          fontsize=6.0, color=INK2)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_ldos_gallery.{ext}"))
    print("wrote fig_ldos_gallery.png/.pdf")


if __name__ == "__main__":
    main()

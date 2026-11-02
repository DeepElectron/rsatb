"""Regenerate the tutorial figures from real RSATB runs.

The figures in docs/assets/figures/ are not drawn by hand: each one is a plot
of what a tutorial input actually produced. To refresh them:

    rsatb run examples/tutorials/graphene_dos.json
    rsatb run examples/tutorials/graphene_transport.json
    rsatb run examples/tutorials/diamond_optics.json
    rsatb run examples/tutorials/diamond_realspace.json
    python examples/tutorials/strain_sweep.py
    python docs/scripts/make_figures.py

A figure whose run has not been made is skipped with a message. The script
also writes docs/overrides/partials/rs-plate.html, Plate I of the landing
page, from the graphene run.

Colours are the site palette (docs/assets/stylesheets/rsatb.css): ink for
structure, ink-blue for RSATB data, vermilion for references and marks, and
ramps between paper and those two for maps and families of curves.
"""

import glob
import json
import pathlib
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
OUT = ROOT / "docs" / "assets" / "figures"
PLATE = ROOT / "docs" / "overrides" / "partials" / "rs-plate.html"

PAPER = "#faf8f3"
INK = "#1c1b19"
MUTED = "#6f6a60"
RULE = "#e3ddd0"
ACCENT = "#2c4a8a"      # ink-blue: RSATB data
WARM = "#b8432a"        # vermilion: references, marks
# Ramps for maps and families of curves, paper -> ink-blue and paper -> vermilion -> ink.
BLUES = LinearSegmentedColormap.from_list("rs_blues", [PAPER, "#9fb0d3", ACCENT, "#14223f"])
EMBER = LinearSegmentedColormap.from_list("rs_ember", [PAPER, "#e6b7a4", WARM, "#3a1a10"])
FAMILY = LinearSegmentedColormap.from_list("rs_family", ["#9fb0d3", ACCENT, "#14223f"])
XX = 0

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": RULE,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "figure.dpi": 100,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def _results(path):
    path = pathlib.Path(path)
    if not (path / "summary.json").exists() and not glob.glob(str(path / "*summary.json")):
        print("skip: no run in", path.relative_to(ROOT))
        return None
    summary_file = sorted(path.glob("*summary.json"))[0]
    return json.loads(summary_file.read_text())


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def _gaussian(x, centres, fwhm, weights=None):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    out = np.zeros_like(x)
    w = np.ones_like(centres) if weights is None else weights
    for chunk in range(0, len(centres), 4096):
        c = centres[chunk:chunk + 4096]
        out += (w[chunk:chunk + 4096, None] *
                np.exp(-0.5 * ((x[None, :] - c[:, None]) / sigma) ** 2)).sum(axis=0)
    return out / (sigma * np.sqrt(2.0 * np.pi))


def graphene_exact(n_cell, t, energy, fwhm):
    """Exact nearest-neighbour graphene DOS at the k-points a Gamma-point
    calculation of an n x n supercell samples, broadened like the KPM run."""
    a1 = np.array([2.4595, 0.0])
    a2 = np.array([1.22975, 2.13])
    pos_b = np.array([0.0, 1.42])
    deltas = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            d = pos_b + i * a1 + j * a2
            if np.linalg.norm(d) < 1.6:
                deltas.append(d)
    deltas = np.array(deltas)
    A = np.array([a1, a2])
    B = 2.0 * np.pi * np.linalg.inv(A).T
    idx = np.arange(n_cell) / n_cell
    kx, ky = np.meshgrid(idx, idx, indexing="ij")
    k = kx.reshape(-1, 1) * B[0] + ky.reshape(-1, 1) * B[1]
    f = np.exp(1j * k @ deltas.T).sum(axis=1)
    e = np.concatenate([abs(t) * np.abs(f), -abs(t) * np.abs(f)])
    return _gaussian(energy, e, fwhm) / len(e)


def fig_graphene_dos():
    run = EXAMPLES / "tutorials" / "results" / "graphene_dos"
    summary = _results(run)
    if summary is None:
        return
    emin, emax = summary["setup"]["energy_bounds"]
    n_cell = summary["input"]["structure"]["supercell"][0]
    tasks = {t["label"]: t for t in summary["input"]["tasks"] if t["type"] == "dos"}

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.1))
    spark = None
    for ax, label in zip(axes, ("coarse", "fine")):
        d = np.load(run / "{}.npz".format(label))
        e, rho = d["energy"], d["dos"][0] / (emax - emin)
        exact = graphene_exact(n_cell, 2.7, e, tasks[label]["dE"])
        ax.fill_between(e, rho, color=ACCENT, alpha=0.08, lw=0)
        ax.plot(e, exact, color=WARM, lw=2.6, alpha=0.45, label="exact, same k-points")
        ax.plot(e, rho, color=ACCENT, lw=1.0, label="RSATB")
        ax.set(xlim=(emin, emax), ylim=(0, rho.max() * 1.18), xlabel="Energy (eV)",
               title='"{}": dE = {} eV'.format(label, tasks[label]["dE"]))
        ax.legend(loc="upper right", fontsize=7.5)
        if label == "coarse":
            ax.set_ylabel("DOS (states / eV / orbital)")
            for x in (-2.7, 2.7):
                ax.annotate("van Hove", (x, rho.max() * 1.02), ha="center", va="bottom",
                            fontsize=7.5, color=MUTED)
            ax.annotate("Dirac point", (0, 0.002), xytext=(0, rho.max() * 0.45), ha="center",
                        fontsize=7.5, color=MUTED,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
            spark = (e, rho)
    fig.tight_layout(w_pad=2.5)
    _save(fig, "tut_graphene_dos.png")
    _plate(*spark, emin, emax)


def _plate(e, rho, emin, emax, t=2.7):
    """Landing-page Plate I: the graphene DOS as an engraved ink line.

    Drawn as SVG from the run itself. Behind it, faintly, the Chebyshev
    polynomials T_1..T_9 on the same (scaled) energy axis -- the functions the
    curve is assembled from. Annotations sit at the computed features: the van
    Hove peaks at +-|t| and the Dirac point. Colours and type come from the
    stylesheet (classes), so the plate follows the light and dark schemes.
    """
    W, x0, x1, base, top = 1000.0, 50.0, 950.0, 300.0, 64.0

    def X(energy):
        return x0 + (energy - emin) / (emax - emin) * (x1 - x0)

    def Y(value):
        return base - value / rho.max() * (base - top)

    def poly(xs, ys):
        return " ".join("{:.1f},{:.1f}".format(a, b) for a, b in zip(xs, ys))

    parts = []
    # Chebyshev ghosts: T_n(x) = cos(n arccos x) on the scaled axis.
    xs = np.linspace(-1.0, 1.0, 400)
    mid, amp = (base + top) / 2.0, (base - top) / 2.0 * 0.92
    for n in range(1, 10):
        tn = np.cos(n * np.arccos(xs))
        ex = emin + (xs + 1.0) / 2.0 * (emax - emin)
        parts.append('<polyline class="cheb" points="{}"/>'.format(poly(X(ex), mid - amp * tn)))

    step = max(1, len(e) // 700)
    line = poly(X(e[::step]), Y(rho[::step]))
    parts.append('<polygon class="dos-fill" points="{:.1f},{:.1f} {} {:.1f},{:.1f}"/>'.format(
        X(e[0]), base, line, X(e[-1]), base))
    parts.append('<polyline class="dos" points="{}"/>'.format(line))
    parts.append('<line class="axis" x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}"/>'.format(
        x0, base, x1, base))
    for tick in (-8, -4, 0, 4, 8):
        parts.append('<line class="axis" x1="{0:.1f}" y1="{1:.1f}" x2="{0:.1f}" y2="{2:.1f}"/>'
                     '<text class="tick" x="{0:.1f}" y="{3:.1f}" text-anchor="middle">{4}</text>'.format(
                         X(tick), base, base + 6, base + 24, "{:+d}".format(tick) if tick else "0"))
    parts.append('<text class="tick" x="{:.1f}" y="{:.1f}" text-anchor="end">energy / eV</text>'.format(
        x1, base + 46))

    # Van Hove peaks: the maximum of the curve near -|t| and +|t|.
    for sign, anchor, dx in ((-1, "end", -18), (1, "start", 18)):
        window = np.abs(e - sign * t) < 0.4
        i = np.flatnonzero(window)[np.argmax(rho[window])]
        px, py = X(e[i]), Y(rho[i])
        lx, ly = px + dx * 3.2, py - 18
        parts.append('<circle class="mark" cx="{:.1f}" cy="{:.1f}" r="3"/>'.format(px, py))
        parts.append('<polyline class="lead" points="{:.1f},{:.1f} {:.1f},{:.1f} {:.1f},{:.1f}"/>'.format(
            px, py - 5, px + dx, ly, lx, ly))
        parts.append('<text class="ann" x="{:.1f}" y="{:.1f}" text-anchor="{}">van Hove singularity</text>'
                     '<text class="ann-sub" x="{:.1f}" y="{:.1f}" text-anchor="{}">E = {}|t|</text>'.format(
                         lx + (6 if sign > 0 else -6), ly + 5, anchor,
                         lx + (6 if sign > 0 else -6), ly + 22, anchor, "+" if sign > 0 else "−"))

    # Dirac point: the zero of the DOS at E = 0. Labelled below the axis,
    # where nothing else is drawn, clear of the "0" tick.
    dx0 = X(0.0)
    ly = base + 52
    parts.append('<circle class="mark" cx="{:.1f}" cy="{:.1f}" r="3"/>'.format(dx0, base))
    parts.append('<polyline class="lead" points="{:.1f},{:.1f} {:.1f},{:.1f} {:.1f},{:.1f}"/>'.format(
        dx0 + 3, base + 4, dx0 + 34, ly - 5, dx0 + 60, ly - 5))
    parts.append('<text class="ann" x="{:.1f}" y="{:.1f}">Dirac point</text>'
                 '<text class="ann-sub" x="{:.1f}" y="{:.1f}">E = 0, the density vanishes linearly</text>'.format(
                     dx0 + 66, ly, dx0 + 66, ly + 17))

    PLATE.parent.mkdir(parents=True, exist_ok=True)
    PLATE.write_text(
        '<svg class="rs-plate" viewBox="0 0 {:.0f} 380" role="img" '
        'aria-label="Density of states of graphene, with van Hove singularities and the Dirac point">'
        '{}</svg>\n'.format(W, "".join(parts)))
    print("wrote", PLATE.relative_to(ROOT))


def fig_graphene_transport():
    run = EXAMPLES / "tutorials" / "results" / "graphene_transport"
    summary = _results(run)
    if summary is None:
        return
    emin, emax = summary["setup"]["energy_bounds"]
    dos = np.load(run / "dos.npz")
    dc = np.load(run / "dc_spectrum.npz")
    msd = np.load(run / "msd.npz")

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))
    ax = axes[0]
    ax.plot(dos["energy"], dos["dos"][0] / (emax - emin), color=ACCENT, lw=1.1)
    ax.set(xlabel="Energy (eV)", ylabel="DOS (states / eV / orbital)", title="Density of states",
           xlim=(-8.5, 8.5), ylim=(0, None))

    ax = axes[1]
    ax.plot(dc["energy"], dc["sigma"][0, :, XX] * 1e-6, color=ACCENT, lw=1.1)
    ax.set(xlabel="Energy (eV)", ylabel=r"$\sigma_{xx}$ (MS, sheet)", title="DC conductivity, T = 0",
           xlim=(-8.5, 8.5), ylim=(0, None))

    ax = axes[2]
    t, E, D = msd["time_fs"], msd["energy"], msd["diffusion"][0]
    cmap = FAMILY
    for i, target in enumerate((0.5, 1.0, 2.0, 2.7)):
        j = int(np.argmin(abs(E - target)))
        ax.plot(t, D[:, j] * 0.1, lw=1.2, color=cmap(0.05 + 0.3 * i),
                label="E = {:.1f} eV".format(E[j]))
    ax.set(xlabel="Time (fs)", ylabel=r"$D(E,t)$ (cm$^2$/s)", title="Diffusion: ballistic, clean",
           xlim=(0, t[-1]), ylim=(0, None))
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(w_pad=2.2)
    _save(fig, "tut_graphene_transport.png")


def fig_strain():
    path = EXAMPLES / "tutorials" / "strain_sweep.npz"
    if not path.exists():
        print("skip: no", path.relative_to(ROOT))
        return
    d = np.load(path)
    strains = d["strains"]
    cmap = FAMILY
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, s in enumerate(strains):
        ax.plot(d["energy_{}".format(i)], d["dos_{}".format(i)], lw=1.1,
                color=cmap(0.1 + 0.8 * i / max(1, len(strains) - 1)),
                label="{:.1f} %".format(100 * s))
    ax.set(xlabel="Energy (eV)", ylabel="DOS (states / eV / orbital)", xlim=(-8.8, 8.8),
           ylim=(0, None))
    ax.legend(title="biaxial strain", fontsize=8, title_fontsize=8, loc="upper left", ncol=1)
    _save(fig, "tut_strain_sweep.png")


def fig_diamond_optics():
    run = EXAMPLES / "tutorials" / "results" / "diamond_optics"
    summary = _results(run)
    if summary is None:
        return
    emin, emax = summary["setup"]["energy_bounds"]
    dos = np.load(run / "dos.npz")
    ac = np.load(run / "ac.npz")
    mu = [t for t in summary["input"]["tasks"] if t["type"] == "ac"][0]["mu"]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))
    ax = axes[0]
    e = dos["energy"] - mu
    rho = dos["dos"][0] / (emax - emin)
    keep = (e > -25) & (e < 15)
    # A Gamma-point 4x4x4 cell has a discrete spectrum; broaden for display
    # (as examples/plot_dos_ac.py does) and keep the raw comb underneath.
    smooth = _gaussian(e[keep], e[keep], 0.6, weights=rho[keep]) * float(np.mean(np.diff(e)))
    ax.plot(e[keep], rho[keep], color=MUTED, lw=0.5, alpha=0.35, label="raw, dE = 0.14 eV")
    ax.fill_between(e[keep], smooth, color=ACCENT, alpha=0.10, lw=0)
    ax.plot(e[keep], smooth, color=ACCENT, lw=1.2, label="broadened, 0.6 eV")
    ax.axvline(0, color=WARM, lw=0.8, ls="--")
    ax.set(xlabel=r"$E - \mu$ (eV)", ylabel="DOS (states / eV / orbital)",
           title="Density of states", ylim=(0, smooth.max() * 1.3), xlim=(-25, 15))
    ax.legend(fontsize=7, loc="upper left")

    w = ac["omega_cm1"] / 8065.56
    sig = ac["sigma"][0, :, XX]
    keep = (w > 0.05) & (w < 30)
    ax = axes[1]
    ax.plot(w[keep], sig.real[keep] * 1e-6, color=ACCENT, lw=1.1, label=r"Re $\sigma_{xx}$")
    ax.plot(w[keep], sig.imag[keep] * 1e-6, color=WARM, lw=1.0, alpha=0.8, label=r"Im $\sigma_{xx}$")
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.set(xlabel=r"$\hbar\omega$ (eV)", ylabel=r"$\sigma$ (MS/m)", title="Optical conductivity")
    ax.legend(fontsize=7.5)

    ax = axes[2]
    eps = ac["epsilon"][0, :, XX]
    ax.plot(w[keep], eps.real[keep], color=ACCENT, lw=1.1, label=r"Re $\varepsilon$")
    ax.plot(w[keep], eps.imag[keep], color=WARM, lw=1.0, alpha=0.8, label=r"Im $\varepsilon$")
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.set(xlabel=r"$\hbar\omega$ (eV)", ylabel=r"$\varepsilon(\omega)$", title="Dielectric function")
    ax.legend(fontsize=7.5)
    fig.tight_layout(w_pad=2.2)
    _save(fig, "tut_diamond_optics.png")


def _gather(run, label, key):
    parts = [np.load(p) for p in sorted(glob.glob(str(run / "{}.rank*.npz".format(label))))]
    if not parts:
        return None, None, None
    shape = tuple(int(n) for n in parts[0]["shape"])
    first = parts[0][key]
    if first.ndim == 1:
        flat = np.zeros(int(np.prod(shape)))
        for p in parts:
            flat[p["point_index"]] = p[key]
        return flat.reshape(shape), shape, parts[0]
    flat = np.zeros((first.shape[0], int(np.prod(shape))))
    for p in parts:
        flat[:, p["point_index"]] = p[key]
    return flat.reshape((-1,) + shape), shape, parts[0]


def fig_diamond_realspace():
    run = EXAMPLES / "tutorials" / "results" / "diamond_realspace"
    summary = _results(run)
    if summary is None:
        return
    rho, shape, _ = _gather(run, "density", "density")
    ldos, _, part = _gather(run, "ldos", "ldos")
    if rho is None or ldos is None:
        print("skip: real-space pieces missing in", run.relative_to(ROOT))
        return
    energies = part["energy"]

    # Grid point (i, j, k) sits at fractional (i/n0, j/n1, k/n2) of the
    # supercell. Plot the k = 0 lattice plane in Cartesian coordinates, so the
    # sheared cell is drawn as the parallelogram it is.
    import ase.io
    structure = summary["input"]["structure"]
    cell = np.asarray(ase.io.read(structure["file"]).cell) * np.asarray(structure["supercell"])[:, None]
    n0, n1, _ = shape
    i = np.arange(n0 + 1)[:, None] / n0
    j = np.arange(n1 + 1)[None, :] / n1
    X = i * cell[0, 0] + j * cell[1, 0]
    Y = i * cell[0, 1] + j * cell[1, 1]

    def plane(values):  # close the parallelogram periodically
        v = values[:, :, 0]
        return np.pad(v, ((0, 1), (0, 1)), mode="wrap")

    picks = [(0, "valence band"), (len(energies) // 2, "band gap"),
             (len(energies) - 1, "conduction band")]
    fig, axes = plt.subplots(1, 1 + len(picks), figsize=(11.0, 2.75))
    panels = [(axes[0], plane(rho), EMBER, r"density (e/Å$^3$)")]
    for ax, (idx, where) in zip(axes[1:], picks):
        panels.append((ax, plane(ldos[idx]), BLUES,
                       "LDOS, {:.0f} eV · {}".format(energies[idx], where)))
    for ax, values, cmap, title in panels:
        im = ax.pcolormesh(X, Y, values, cmap=cmap, shading="gouraud", rasterized=True)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7)
    fig.tight_layout(w_pad=0.6)
    _save(fig, "tut_diamond_realspace.png")


if __name__ == "__main__":
    wanted = set(sys.argv[1:])
    for name, fn in (("graphene_dos", fig_graphene_dos),
                     ("graphene_transport", fig_graphene_transport),
                     ("strain", fig_strain),
                     ("diamond_optics", fig_diamond_optics),
                     ("diamond_realspace", fig_diamond_realspace)):
        if not wanted or name in wanted:
            fn()

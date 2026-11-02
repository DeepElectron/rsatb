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


def fig_moire(theta_deg=3.48, half_width=120.0):
    """Landing-page hero: twisted bilayer graphene, drawn bond by bond.

    Pure geometry, no calculation: two graphene sheets (C-C 1.42 A) rotated
    by +-theta/2. The moire pattern -- AA and AB stacking regions with period
    a / (2 sin(theta/2)) -- emerges from the overlay of the two bond networks.
    Rendered on a transparent background inside a circular field of view.
    """
    from matplotlib.collections import LineCollection

    a_cc = 1.42
    a1 = np.array([np.sqrt(3.0) * a_cc, 0.0])
    a2 = np.array([np.sqrt(3.0) * a_cc / 2.0, 1.5 * a_cc])
    basis = [np.zeros(2), np.array([0.0, a_cc])]
    deltas = [np.array([0.0, a_cc]),
              np.array([np.sqrt(3.0) * a_cc / 2.0, -a_cc / 2.0]),
              np.array([-np.sqrt(3.0) * a_cc / 2.0, -a_cc / 2.0])]
    n = int(half_width / a_cc) + 4
    ij = np.array([(i, j) for i in range(-n, n + 1) for j in range(-n, n + 1)], dtype=float)
    sites_a = ij[:, :1] * a1 + ij[:, 1:] * a2 + basis[0]

    def layer(angle):
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        segs = []
        for d in deltas:          # every bond once: from each A site to its 3 B neighbours
            start = sites_a @ rot.T
            end = (sites_a + d) @ rot.T
            segs.append(np.stack([start, end], axis=1))
        segs = np.concatenate(segs)
        mid = segs.mean(axis=1)
        return segs[np.hypot(mid[:, 0], mid[:, 1]) < half_width]

    from matplotlib.colors import to_rgb
    from PIL import Image

    theta = np.radians(theta_deg)
    fig = plt.figure(figsize=(7.0, 7.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.axis("off")
    for segs, colour in ((layer(-theta / 2), INK), (layer(theta / 2), ACCENT)):
        # A soft circular field of view: full strength at the centre, fading out.
        r = np.hypot(*segs.mean(axis=1).T) / half_width
        fade = np.clip((1.0 - r) / 0.35, 0.0, 1.0) ** 1.5
        rgba = np.zeros((len(segs), 4))
        rgba[:, :3] = to_rgb(colour)
        rgba[:, 3] = 0.9 * fade
        ax.add_collection(LineCollection(segs, colors=rgba, linewidths=0.42))
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_width, half_width)
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "home_moire.png"
    fig.savefig(png, dpi=160, transparent=True, facecolor="none", bbox_inches=None)
    plt.close(fig)
    # Ship a compact WebP; the PNG is only an intermediate.
    Image.open(png).save(OUT / "home_moire.webp", "WEBP", quality=74, method=6)
    png.unlink()
    period = 2.46 / (2 * np.sin(theta / 2))
    print("wrote", (OUT / "home_moire.webp").relative_to(ROOT),
          "theta={:.2f} deg, moire period {:.1f} A".format(theta_deg, period))


DEVICE = ROOT / "docs" / "overrides" / "partials" / "rs-device.html"


def _jittered(xmin, xmax, ymin, ymax, step, jitter, rng, keep=0.85):
    """Amorphous sites: a grid with random displacements and random vacancies."""
    xs, ys = np.meshgrid(np.arange(xmin, xmax, step), np.arange(ymin, ymax, step))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    pts = pts + rng.uniform(-jitter, jitter, pts.shape)
    return pts[rng.random(len(pts)) < keep]


def fig_device(half=60.0, px=0.12, seed=7):
    """Landing-page hero, second specimen: a gate-all-around Si transistor.

    A geometric model for illustration, not a calculation, rendered the way such
    structures are imaged: as an atomic-resolution Z-contrast (HAADF-STEM-like)
    picture, where each projected atomic column is a spot of brightness ~ Z^1.7.
    The crystalline Si channel is viewed along [110] (dumbbell columns 1.36 A
    apart); As dopants in source and drain are brighter columns; the SiO2 gate
    oxide and SiN spacers are amorphous speckle; the W gate and the contacts are
    dense crystalline lattices. Transparent background, circular field of view,
    labels as an SVG overlay that uses the site's fonts and colours.
    """
    from scipy.ndimage import gaussian_filter
    from matplotlib.colors import to_rgb
    from PIL import Image

    rng = np.random.default_rng(seed)
    a = 5.431
    ex = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)   # along the channel
    ey = np.array([0.0, 0.0, 1.0])                   # across the channel
    fcc = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    basis = np.vstack([fcc, fcc + 0.25]) * a
    n = int(half / a * 1.6) + 3
    cells = np.array([(i, j, k) for i in range(-n, n + 1) for j in range(-n, n + 1)
                      for k in range(-3, 4)], dtype=float) * a
    atoms = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    cols = np.unique(np.round(np.stack([atoms @ ex, atoms @ ey], axis=1), 2), axis=0)

    t_si, gate_half, spacer = 13.6, 27.0, 9.0
    ox_top = t_si + 8.4
    in_box = (np.abs(cols[:, 0]) < half + 2) & (np.abs(cols[:, 1]) <= t_si)
    si = cols[in_box]
    w_si = np.ones(len(si))
    dop = (np.abs(si[:, 0]) > gate_half + 3) & (rng.random(len(si)) < 0.045)
    w_si[dop] = 3.4                                   # As: (33/14)^1.7 ~ 4.3, softened

    sites, weights = [si], [w_si]
    for sign in (1, -1):
        # Gate oxide (amorphous SiO2) under the gate.
        ox = _jittered(-gate_half, gate_half, t_si + 0.8, ox_top, 1.25, 0.5, rng)
        ox[:, 1] *= sign
        sites.append(ox); weights.append(rng.choice([0.55, 0.22], len(ox), p=[0.35, 0.65]))
        # Nitride spacers beside the gate.
        for side in (1, -1):
            sp = _jittered(gate_half + 0.6, gate_half + spacer, t_si + 0.8, half + 2, 1.3, 0.5, rng)
            sp[:, 0] *= side; sp[:, 1] *= sign
            sites.append(sp); weights.append(np.full(len(sp), 0.32))
        # W gate: a dense square lattice of bright columns.
        gx, gy = np.meshgrid(np.arange(-gate_half + 0.8, gate_half, 1.58),
                             np.arange(ox_top + 1.0, half + 2, 1.58))
        g = np.stack([gx.ravel(), sign * gy.ravel()], axis=1)
        sites.append(g); weights.append(np.full(len(g), 1.9))
        # Contacts above source and drain: a crystalline metal grain, rotated.
        for side in (1, -1):
            ang = np.radians(17 * side * sign)
            c, s = np.cos(ang), np.sin(ang)
            u, v = np.meshgrid(np.arange(-40, 40, 2.05), np.arange(-40, 40, 2.05 * np.sqrt(3) / 2))
            u = u + (np.arange(u.shape[0])[:, None] % 2) * 1.025
            p = np.stack([u.ravel() * c - v.ravel() * s, u.ravel() * s + v.ravel() * c], axis=1)
            p = p + np.array([side * (gate_half + spacer + 20), sign * (t_si + 22)])
            m = (side * p[:, 0] > gate_half + spacer + 0.6) & (sign * p[:, 1] > t_si + 0.8)
            sites.append(p[m]); weights.append(np.full(m.sum(), 1.35))

    pts = np.vstack(sites)
    w = np.concatenate(weights)
    npx = int(round(2 * half / px))
    img = np.zeros((npx, npx))
    ix = np.round((pts[:, 0] + half) / px).astype(int)
    iy = np.round((half - pts[:, 1]) / px).astype(int)
    ok = (ix >= 0) & (ix < npx) & (iy >= 0) & (iy < npx)
    np.add.at(img, (iy[ok], ix[ok]), w[ok])
    img = gaussian_filter(img, 0.36 / px)
    img /= np.percentile(img[img > 0], 99.7)
    img = np.clip(img, 0, 1)

    yy, xx = np.mgrid[0:npx, 0:npx]
    r = np.hypot(xx - npx / 2, yy - npx / 2) / (npx / 2)
    fade = np.clip((1.0 - r) / 0.38, 0.0, 1.0) ** 1.5
    lo, hi = np.array(to_rgb(ACCENT)), np.array(to_rgb(INK))
    rgb = lo[None, None, :] * (1 - img[..., None]) + hi[None, None, :] * img[..., None]
    alpha = (img ** 0.85) * fade
    rgba = np.concatenate([rgb, alpha[..., None]], axis=2)
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(
        OUT / "home_device.webp", "WEBP", quality=78, method=6)

    # Labels: an SVG overlay in the same frame, 10 units per Angstrom.
    S = 10.0

    def X(v):
        return (v + half) * S

    def Y(v):
        return (half - v) * S

    parts = []

    def label(text, ax_, ay_, tx, ty, anchor="start"):
        parts.append('<polyline class="lead" points="{:.0f},{:.0f} {:.0f},{:.0f}"/>'
                     '<circle class="pin" cx="{:.0f}" cy="{:.0f}" r="5"/>'
                     '<text class="lbl" x="{:.0f}" y="{:.0f}" text-anchor="{}">{}</text>'.format(
                         X(ax_), Y(ay_), X(tx), Y(ty), X(ax_), Y(ay_),
                         X(tx) + (10 if anchor == "start" else -10), Y(ty) + 8, anchor, text))

    label("W gate", 6, 40, 20, 48)
    label('SiO<tspan class="sub" dy="6">2</tspan>', -12, t_si + 4, -30, 34, "end")
    label("SiN spacer", -31, 38, -40, 46, "end")
    label("Si channel [110]", 4, 3, 18, 8)
    label("contact", 46, 24, 56, 36, "end")
    d = si[dop]
    d0 = d[np.argmin(np.hypot(d[:, 0] - 40, d[:, 1] + 6))]
    label("As dopant", d0[0], d0[1], 54, -30, "end")
    label("source", -44, -8, -46, -26, "end")
    parts.append('<line class="bar" x1="{:.0f}" y1="{:.0f}" x2="{:.0f}" y2="{:.0f}"/>'
                 '<text class="lbl" x="{:.0f}" y="{:.0f}" text-anchor="middle">1 nm</text>'.format(
                     X(-22), Y(-43), X(-12), Y(-43), X(-17), Y(-43) - 14))
    W = 2 * half * S
    DEVICE.parent.mkdir(parents=True, exist_ok=True)
    DEVICE.write_text('<svg class="rs-device-labels" viewBox="0 0 {0:.0f} {0:.0f}" aria-hidden="true">{1}</svg>\n'
                      .format(W, "".join(parts)))
    kb = (OUT / "home_device.webp").stat().st_size / 1024
    print("wrote", (OUT / "home_device.webp").relative_to(ROOT), "{:.0f} kB,".format(kb),
          len(si), "Si columns,", int(dop.sum()), "dopants;", DEVICE.relative_to(ROOT))


if __name__ == "__main__":
    wanted = set(sys.argv[1:])
    for name, fn in (("moire", fig_moire),
                     ("device", fig_device),
                     ("graphene_dos", fig_graphene_dos),
                     ("graphene_transport", fig_graphene_transport),
                     ("strain", fig_strain),
                     ("diamond_optics", fig_diamond_optics),
                     ("diamond_realspace", fig_diamond_realspace)):
        if not wanted or name in wanted:
            fn()

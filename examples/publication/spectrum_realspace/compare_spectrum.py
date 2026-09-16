"""Stochastic DOS/PDOS against diagonalization, with the two error sources split.

Both sides are the SAME estimator: identical moment count, Jackson kernel,
edge inset, recurrence and 2*rho amplitude convention (kpm_ref.py mirrors
rsatb/calc/kpm.hpp). They differ only in how the Chebyshev moments are
obtained -- stochastically traced by rsatb, summed exactly over diagonalized
eigenvalues here. So a deviation is never "two methods disagreeing"; it is
the trace estimate, plus whatever physical difference exists between the two
spectra being compared. Hence two columns:

  same-cell : reference = the SAME supercell the stochastic run used (its
              exact spectrum, k-folded). Isolates the ESTIMATOR error.
  converged : reference = the dense-mesh limit. Adds the physical finite-cell
              error, and is what the figure quotes.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from kpm_ref import kpm_curves, n_moments_for

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
R = lambda *p: __import__("os").path.join(HERE, "results", *p)
FW = 0.2
SHELLS_2S2P1D = {"s": [0, 1], "p": [2, 3, 4, 5, 6, 7], "d": [8, 9, 10, 11, 12]}
SHELLS_4S4P1D = {"s": [0, 1, 2, 3], "p": list(range(4, 16)), "d": list(range(16, 21))}

SYS = [
    # name, run, dE, (n_basis, n_atom), shells, same-cell ref, converged ref
    ("diamond", "dia_pdos", 0.3, (44928, 3456), SHELLS_2S2P1D,
     ("dia_rs_k12", 44928, 3456), ("dia_k24", 359424, 27648)),
    ("silicon", "si_pdos", 0.3, (44928, 3456), SHELLS_2S2P1D,
     ("si_rs_k12", 44928, 3456), ("si_k24", 359424, 27648)),
    ("aluminum", "al_pdos", 0.3, (36288, 1728), SHELLS_4S4P1D,
     ("al_rs_k12", 36288, 1728), ("al_k24", 290304, 13824)),
]


def curves(run, de, sizes, shells, ref):
    z = np.load(R(run, "pdos_slot.npz"))
    e = z["energy"]
    de_grid = e[1] - e[0]
    scale = e[e.size // 2] - e[0]
    nb, na = sizes
    to_atom = nb / (2.0 * scale) / na
    ours = np.vstack([z["total"][0][None, :],
                      np.stack([z["pdos"][0][s].sum(0) for s in shells.values()])]) * to_atom

    kdir, ref_nb, ref_na = ref
    eps = np.fromfile(R(kdir, "k_eigenvalues.f64"))
    w = np.fromfile(R(kdir, "k_lowdin_weight_slot.f64")).reshape(len(sum(shells.values(), [])), -1)
    rows = np.vstack([np.ones((1, eps.size))] + [w[s].sum(0)[None, :] for s in shells.values()])
    refc = kpm_curves(eps, rows, e, n_moments_for(e, de), ref_nb) * (ref_nb / (2.0 * scale) / ref_na)
    sm = lambda y: gaussian_filter1d(y, (FW / 2.3548) / de_grid, mode="nearest")
    return np.stack([sm(r) for r in ours]), np.stack([sm(r) for r in refc])


print(f"{'system':<10} {'reference':<18} {'DOS':>7} {'s':>7} {'p':>7} {'d':>7}")
for name, run, de, sizes, shells, same, conv in SYS:
    for label, ref in (("same-cell (exact)", same), ("converged (dense k)", conv)):
        ours, refc = curves(run, de, sizes, shells, ref)
        peak = refc[0].max()
        dv = [100 * np.sqrt(((o - r) ** 2).mean()) / peak for o, r in zip(ours, refc)]
        print(f"{name if label.startswith('same') else '':<10} {label:<18}"
              + "".join(f"{d:>6.2f}%" for d in dv))
print("\nrms over the plotted window, % of that reference's peak.")

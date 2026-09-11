"""Absolute scale of the transport prefactors.

The FFT-Chebyshev tests lock the dc spectrum against a moment-sum reference
that carries the same constant, so they cannot see a wrong prefactor. This
test starts from the physics instead: on the two-atom diamond cell at Gamma
it evaluates the spin-degenerate Kubo-Greenwood sum over the dense
eigenpairs of the production operators (dumped from the same setup, so the
velocity convention is not re-derived) with the Jackson delta of the run's
moment count, and requires the stochastic dc spectrum to match it in
absolute scale. A missing pi or spin factor fails this by 3x or more; the
stochastic noise at 256 samples on 26 states is a few per cent on the
window integral.
"""
import os

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from mpi4py import MPI  # noqa: F401

from rsatb import (HamiltonianFile, LcaoModel, OrbitalFiles, PreconditionerOptions,
                   RSATB, SetupOptions, Structure)

DATA = os.path.join(os.path.dirname(__file__), "data", "diamond")
E2_OVER_HBAR_ANG = 2.4341348e6  # S/m for velocities in eV Angstrom
G_SPIN = 2.0


def jackson(N):
    D = N + 1.0
    th = np.pi / D
    n = np.arange(1, N)
    g = np.empty(N)
    g[0] = 1.0
    g[1:] = ((D - n) * np.cos(th * n) + np.cos(th) / np.sin(th) * np.sin(th * n)) / D
    return g


def window_from_cheb_grid(E):
    N = E.size
    t0, t1 = np.cos(np.pi * (N - 0.5) / N), np.cos(np.pi * 0.5 / N)
    a = (E[-1] - E[0]) / (t1 - t0)
    return E[0] - a * t0, a


def jackson_delta(E, eps, c, a, N):
    """delta_J(E - eps) in 1/eV, exactly as kpm.hpp sums it."""
    g = jackson(N)
    x = np.clip((E - c) / a, -1 + 1e-12, 1 - 1e-12)
    y = np.clip((eps - c) / a, -1.0, 1.0)
    tx, ty = np.arccos(x)[:, None], np.arccos(y)[None, :]
    out = np.full((E.size, eps.size), g[0])
    for m0 in range(1, N, 256):
        m = np.arange(m0, min(m0 + 256, N))
        out += 2.0 * np.einsum("m,km,mn->kn", g[m], np.cos(tx * m), np.cos(ty.T * m).T)
    return out / (np.pi * np.sqrt(1.0 - x[:, None] ** 2)) / a


def dense(cpp, which, axis=0):
    r, c, v = cpp.dump_operator_coo(which, axis, 0)
    n = int(max(r.max(), c.max())) + 1
    return sp.coo_matrix((v, (r, c)), shape=(n, n)).toarray()


def test_dc_spectrum_has_the_kubo_greenwood_scale():
    model = LcaoModel(hamiltonian=HamiltonianFile(os.path.join(DATA, "hamiltonians.h5")),
                      orbitals=OrbitalFiles(dir=DATA + os.sep, names=["C_gga_7au_100Ry_2s2p1d.orb"]))
    rs = RSATB(Structure.from_file(os.path.join(DATA, "conf.vasp")), model,
               SetupOptions(kpoints=[[0.0, 0.0, 0.0]], nsamples=256,
                            preconditioner=PreconditionerOptions(operator_form="solve")),
               npar=1)
    n_moments = 200
    sigma, energy = rs.compute_dc_spectrum(n_moments)
    sig = (sigma[0][:, 0] + sigma[0][:, 3] + sigma[0][:, 5]) / 3.0

    cpp = rs._cpp_obj
    H, S = dense(cpp, "H"), dense(cpp, "S")
    eps, C = sla.eigh(H, S)
    Sinv = np.linalg.inv(S)
    W = np.zeros((eps.size, eps.size))
    for ax in range(3):
        V = 1j * (dense(cpp, "bond", ax) - dense(cpp, "xbar", ax) @ Sinv @ H
                  + H @ Sinv @ dense(cpp, "delta", ax))
        Ve = C.conj().T @ V @ C
        W += np.abs(Ve) ** 2 / 3.0
    cell = np.array(cpp.cell()).reshape(3, 3)
    omega = abs(np.linalg.det(cell))
    c, a = window_from_cheb_grid(energy)
    dJ = jackson_delta(energy, eps, c, a, n_moments)
    ref = np.pi * G_SPIN * E2_OVER_HBAR_ANG / omega * np.einsum("kn,nm,km->k", dJ, W, dJ)

    sel = ref > 0.05 * ref.max()
    ratio = np.trapezoid(sig[sel], energy[sel]) / np.trapezoid(ref[sel], energy[sel])
    assert abs(ratio - 1.0) < 0.15, f"dc spectrum / Kubo-Greenwood = {ratio:.3f}"
    corr = np.corrcoef(sig[sel], ref[sel])[0, 1]
    assert corr > 0.99, f"shape correlation {corr:.4f}"

#!/usr/bin/env python3
"""Exact-diagonalization ingredients for the transport reference curves.

For each system: build the SAME operators the stochastic estimator uses
(via the dump_operator_coo binding -- no convention re-derivation), solve
the dense generalized problem H C = S C eps, rotate the commutator
velocity V_a = i (bond_a - xbar_a S^-1 H + H S^-1 delta_a) into the
eigenbasis, and store

    eps    (N,)      eigenvalues, eV
    Wabc   (N, N)    mean over a=x,y,z of |<n|v_a|m>|^2   (abc panels)
    Wx     (N, N)    |<n|v_x|m>|^2                        (msd axis x)

Everything downstream (sigma(E), sigma(omega), D(E,t) at absolute scale)
is a cheap kernel sum over these. S^-1 = C C^H in the eigenbasis, so the
composite needs no explicit inverse: V_eig = i (Bt - Xt E + E Dt) with
Bt = C^H B C etc. and E = diag(eps).
"""
from mpi4py import MPI  # noqa: F401  (initializes MPI for the extension)
import os
import sys
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla

from rsatb.inputs.check import load_and_resolve
from rsatb.inputs.build import build_calculation

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEMS = {"diamond": "diamond.json", "silicon": "silicon.json", "aluminum": "aluminum.json"}


def dense(cpp, which, axis=0):
    r, c, v = cpp.dump_operator_coo(which, axis, 0)
    n = int(max(r.max(), c.max())) + 1
    return sp.coo_matrix((v, (r, c)), shape=(n, n)).toarray()


def run(name):
    resolved, _ = load_and_resolve(f"{HERE}/{SYSTEMS[name]}")
    # the sandwich form releases S after building Z; solve keeps it, and
    # the operators themselves are form-independent
    resolved["run"]["preconditioner"]["operator_form"] = "solve"
    calc, model, options = build_calculation(resolved, log_level="silent")
    cpp = calc._cpp_obj

    H = dense(cpp, "H")
    S = dense(cpp, "S")
    eps, C = sla.eigh(H, S)
    E = eps  # (N,)
    print(f"[{name}] N={len(eps)} eig {eps.min():.3f}..{eps.max():.3f}", flush=True)

    Wabc = None
    Wx = None
    for a in range(3):
        Bt = C.conj().T @ dense(cpp, "bond", a) @ C
        Xt = C.conj().T @ dense(cpp, "xbar", a) @ C
        Dt = C.conj().T @ dense(cpp, "delta", a) @ C
        V = 1j * (Bt - Xt * E[None, :] + E[:, None] * Dt)
        V = 0.5 * (V + V.conj().T)  # clean the ~1e-8 assembly asymmetry
        W = np.abs(V) ** 2
        if a == 0:
            Wx = W
        Wabc = W / 3.0 if Wabc is None else Wabc + W / 3.0
        print(f"[{name}] axis {a} done, |V| max {np.abs(V).max():.3f}", flush=True)

    cell = np.array(calc.cell() if callable(getattr(calc, "cell", None)) else calc._cpp_obj.cell()).reshape(3, 3)
    np.savez(f"{HERE}/results/ref_{name}_eig.npz", eps=eps, Wabc=Wabc, Wx=Wx,
             cell=cell, norb=len(eps))
    print(f"[{name}] wrote results/ref_{name}_eig.npz  cell_vol={abs(np.linalg.det(cell)):.2f}", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or list(SYSTEMS):
        run(name)

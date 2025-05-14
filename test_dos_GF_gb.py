#!/usr/bin/env python3
"""
Compute and print / save the electronic density‑of‑states (DOS) for a
list of Aluminium super‑cell test‑systems that already live on disk in
VASP / HDF5 form.  The workflow is identical to the previous benchmark
script but, instead of timing the Chebyshev trace build, we actually
Fourier‑transform it onto an energy grid and output ρ(E).

Usage
-----
$ python al_dos.py               # default systems, 20 samples each
$ python al_dos.py -S 50 -d 0.002  

Files written
-------------
DOS_<system>.dat   two‑column ASCII (E, rho)
DOS_<system>.png   quick‐look plot (requires matplotlib)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
import scipy as sp
from ase.io import read
from matplotlib import pyplot as plt  # only used if available at runtime
from scipy.fft import fft, fftshift

from src.data import AtomicData, _keys, OrbitalMapper, block_to_feature
from src.hs.r2k_sparse import R2KSparse
from src.solver.time_GF_gb import (
    _tgf,
    dos,
    dos_real,
    normalize_large_matrix,
    trace_tgf,
)

# ------------------------------------------------------------------ BLAS threads
def set_nthreads(n: int = 8) -> None:
    """Limit OpenMP / MKL usage to *n* threads for reproducibility."""
    os.environ["OMP_NUM_THREADS"] = str(n)


# ------------------------------------------------------------------ helpers
def tlog(msg: str) -> None:
    """Timestamped stdout."""
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}")

# ------------------------------------------------------------------ I/O + workflow

def run_system(
    system: str,
    *,
    sampling_number: int,
    dE: float,
    data_root: pathlib.Path = pathlib.Path("./rsatb/src/test/data"),
) -> None:
    """Load (H,S) for *system*, compute DOS, and save to disk."""

    tlog(f"=== {system}: loading input ===")
    # -- 1. ASE structure (defines cell & species order) ------------------
    at = read(data_root / f"{system}.vasp")
    atomicdata = AtomicData.from_ase(at, r_max=7.4).to_dict()
    atomicdata[_keys.KPOINT_KEY] = np.array([[0.0, 0.2, -0.2]])

    # -- 2. Orbital basis & R→k transform ---------------------------------
    mapper = OrbitalMapper(basis={"Al": "4s4p1d"})
    h2k = R2KSparse(idp=mapper, spin_deg=mapper.spin_deg)
    s2k = R2KSparse(
        idp=mapper,
        spin_deg=mapper.spin_deg,
        edge_field=_keys.EDGE_OVERLAP_KEY,
        node_field=_keys.NODE_OVERLAP_KEY,
        out_field=_keys.OVERLAP_KEY,
    )

    # Block Hamiltonian / overlap from HDF5
    H_block = h5py.File(data_root / f"H_{system}.h5", "r")["0"]
    S_block = h5py.File(data_root / f"S_{system}.h5", "r")["0"]

    block_to_feature(atomicdata, idp=mapper, blocks=H_block, overlap_blocks=S_block)
    atomicdata = h2k(atomicdata)
    atomicdata = s2k(atomicdata)

    H_k = (
        atomicdata[_keys.HAMILTONIAN_KEY][0]
        .to_scipy_sparse()
        .tocsr()
        .astype(np.complex128)
    )
    S_k = (
        atomicdata[_keys.OVERLAP_KEY][0]
        .to_scipy_sparse()
        .tocsr()
        .astype(np.complex128)
    )

    tlog("computing DOS …")
    n = H_k.shape[0]
    v0 = np.random.normal(size=(n, sampling_number)) + 1j * np.random.normal(size=(n, sampling_number))

    rho, E = dos(
        A_csr=H_k,
        B_csr=S_k,
        dE=dE,
        v0=v0,
        verbose=True,
    )

    rho_real, E_real = dos_real(
        A_csr=H_k,
        B_csr=S_k,
        dE=dE,
    )

    # -- 3. save & quick‑look plot ---------------------------------------
    out_dat = pathlib.Path(f"DOS_{system}.dat")
    np.savetxt(out_dat, np.column_stack([E, rho]), header="E   rho(E)")
    tlog(f"DOS → {out_dat.resolve()}")

    try:
        out_png = pathlib.Path(f"DOS_{system}.png")
        plt.figure(figsize=(4, 3))
        plt.plot(E, rho, label="DOS", color="blue")
        plt.plot(E_real, rho_real, label="DOS (real)", color="red", linestyle="--")
        plt.xlabel("Energy (eV)")
        plt.ylabel(r"$\rho(E)$")
        plt.title(system)
        plt.tight_layout()
        plt.legend()
        plt.savefig(out_png, dpi=150)
        plt.close()
        tlog(f"plot  → {out_png.resolve()}")
    except Exception:
        # matplotlib not installed or headless backend
        pass

def main() -> None:
    set_nthreads(8)

    samples = 100000
    dE = 1
    for sys in ["Al"]:#, "Al333", "Al444", "Al555"]:
        tlog(f"\n===== System: {sys} =====")
        run_system(sys, sampling_number=samples, dE=dE)

    tlog("all done!")


if __name__ == "__main__":
    main()

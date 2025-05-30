#!/usr/bin/env python3
import pathlib
import os, time

from matplotlib.pylab import eigvalsh
# ------------------------------------------------------------------ threading
nthreads = 8
os.environ["OMP_NUM_THREADS"] = str(nthreads)

# ------------------------------------------------------------------ imports
from src.data import AtomicData, _keys, OrbitalMapper, block_to_feature
from src.hs.r2k import R2K
from src.hs.r2k_sparse import R2KSparse
from src.solver.time_GF_gb import _tgf, normalize_large_matrix, normalize_small_matrix, random_trace_evaluation, trace_tgf
from ase.io import read
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import h5py

# ------------------------------------------------------------------ helpers
def tlog(msg: str):
    """timestamped print"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def wall(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return time.perf_counter() - t0, out

# ------------------------------------------------------------------ workflow
def run_single(system: str, sampling_number: int, *, t: float = 2.0, tau: float = 1e-14):
    """Return (n, nnz, wall_s) for given (system, S)."""
    # --- load input --------------------------------------------------------
    at = read(f"./rsatb/src/test/data/{system}.vasp")
    atomicdata = AtomicData.from_ase(at, r_max=7.4).to_dict()
    atomicdata[_keys.KPOINT_KEY] = np.array([[0., 0.2, -0.2]])

    idp = OrbitalMapper(basis={"Al": "4s4p1d"})
    tlog("Basis    : 4s4p1d, spin deg = %d" % idp.spin_deg)        ### LOG >

    # ------------------------------------------------------------------ R→k transforms
    h2k_sparse = R2KSparse(idp=idp, spin_deg=idp.spin_deg)
    s2k_sparse = R2KSparse(
        idp=idp, spin_deg=idp.spin_deg,
        edge_field=_keys.EDGE_OVERLAP_KEY,
        node_field=_keys.NODE_OVERLAP_KEY,
        out_field=_keys.OVERLAP_KEY,
    )

    tlog("=== 2. Loading block H/S from disk ===")                   ### LOG >
    H = h5py.File(f"./rsatb/src/test/data/H_{system}.h5", "r")["0"]
    S = h5py.File(f"./rsatb/src/test/data/S_{system}.h5", "r")["0"]

    block_to_feature(atomicdata, idp=idp, blocks=H, overlap_blocks=S)

    tlog("=== 3. R→k sparse transforms ===")                         ### LOG >
    atomicdata = h2k_sparse(atomicdata)
    atomicdata = s2k_sparse(atomicdata)

    H_ = atomicdata[_keys.HAMILTONIAN_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)
    S_ = atomicdata[_keys.OVERLAP_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)

    tlog(f"H size = {H_.shape},  nnz = {H_.nnz:,}")                  ### LOG >
    tlog(f"S size = {S_.shape},  nnz = {S_.nnz:,}")                  ### LOG >

    # ------------------------------------------------------------------ stochastic trace
    tlog("=== 4. Stochastic Chebyshev trace ===")                    ### LOG >
    n = H_.shape[0]
    H_nl, t_nl, shift =  normalize_large_matrix(H_, t, B=S_)
    H_ns, t_ns =  normalize_small_matrix(H_, t)
    t0 = time.perf_counter()

    v0 = np.random.normal(size=(n, sampling_number)) + 1j * np.random.normal(size=(n, sampling_number))
    
    steps_maxl= _tgf(H_nl, t_max=t_nl, B_csr=S_, v0=v0 , trace=True)
    trace_zl = trace_tgf(steps_max=steps_maxl, t=t_nl) / np.e**(1j*shift*t)

    trace_ll = random_trace_evaluation(H_nl, t=t_nl, sampling_number=sampling_number,
                                    tau=tau, rng=None, overlap_matrix=S_, verbose=False) / np.e**(1j*shift*t)
    
    steps_maxs= _tgf(H_ns, t_max=t_ns, B_csr=S_, v0=v0 , trace=True)
    trace_zs = trace_tgf(steps_max=steps_maxs, t=t_ns)

    trace_ls = random_trace_evaluation(H_ns, t=t_ns, sampling_number=sampling_number,
                                    tau=tau, rng=None, overlap_matrix=S_, verbose=False)
    
     
    evals = eigvalsh(H_.toarray())
    trace_true = np.sum(np.exp(-1j * t * evals))

    print(f"trace_t = {trace_true}")
    print(f"trace_zl = {trace_zl}")
    print(f"trace_ll = {trace_ll}")
    print(f"trace_zs = {trace_zs}")
    print(f"trace_ls = {trace_ls}")
    wall_s = time.perf_counter() - t0

    return n, H_.nnz, wall_s

# ---------------- parameter grid -------------------------------------------
SYSTEMS = [
    "Al", "Al333", "Al444", "Al555"]
    #"Al666"]#, "Al777", "Al101010", "Al15151515", "Al20202020",
#]
S_LIST = [1, 10, 100]
CSV_OUT = pathlib.Path("bench_trace_scaling.csv")

# ---------------- run benchmark --------------------------------------------
rows = []
for sys in SYSTEMS:
    tlog(f"System = {sys}")
    for S in S_LIST:
        n, nnz, wt = run_single(sys, S)
        tlog(f"  S={S:6d}   n={n:6d}   time={wt:6.2f}s")
        rows.append({"system": sys, "n": n, "nnz": nnz,
                     "S": S, "wall_s": wt})

# save CSV
pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
print(f"\nCSV saved → {CSV_OUT.resolve()}")

# ---------------- heat‑map plot --------------------------------------------
df = pd.DataFrame(rows)
for sys, g in df.groupby("system"):
    pivot = g.pivot(index="n", columns="S", values="wall_s").sort_index()
    plt.figure(figsize=(5, 4))
    im = plt.imshow(pivot, origin="lower", aspect="auto",
                    extent=[np.log10(pivot.columns.min()), np.log10(pivot.columns.max()),
                            np.log2(pivot.index.min()),  np.log2(pivot.index.max())])
    plt.colorbar(im, label="wall‑time (s)")
    plt.xticks(np.log10(pivot.columns), pivot.columns, rotation=45)
    plt.yticks(np.log2(pivot.index), pivot.index.astype(int))
    plt.title(f"{sys}  |  time vs n,S")
    plt.xlabel("S  (random vectors)")
    plt.ylabel("n  (matrix dim)")
    plt.tight_layout()
    plt.show()
#!/usr/bin/env python3
import os, time
# ------------------------------------------------------------------ threading
nthreads = 8
os.environ["OMP_NUM_THREADS"] = str(nthreads)

# ------------------------------------------------------------------ imports
from src.data import AtomicData, _keys, OrbitalMapper, block_to_feature
from src.hs.r2k import R2K
from src.hs.r2k_sparse import R2KSparse
from src.solver.time_GF_gb import normalize, random_trace_evaluation
from ase.io import read
import numpy as np
import scipy as sp
import h5py

# ------------------------------------------------------------------ helpers
def tlog(msg: str):
    """timestamped print"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ------------------------------------------------------------------ workflow
tlog("=== 1. Building atomic structure ===")                     ### LOG >
sysname = "Al"
at = read(f"./rsatb/src/test/data/{sysname}.vasp")
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
H = h5py.File(f"./rsatb/src/test/data/H_{sysname}.h5", "r")["0"]
S = h5py.File(f"./rsatb/src/test/data/S_{sysname}.h5", "r")["0"]

block_to_feature(atomicdata, idp=idp, blocks=H, overlap_blocks=S)

tlog("=== 3. R→k sparse transforms ===")                         ### LOG >
atomicdata = h2k_sparse(atomicdata)
atomicdata = s2k_sparse(atomicdata)

H_ = atomicdata[_keys.HAMILTONIAN_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)
S_ = atomicdata[_keys.OVERLAP_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)

tlog(f"H size = {H_.shape},  nnz = {H_.nnz:,}")                  ### LOG >
tlog(f"S size = {S_.shape},  nnz = {S_.nnz:,}")                  ### LOG >

# ------------------------------------------------------------------ exact trace (small n)
tlog("=== 4. Exact trace via dense eig(H,S) ===")                ### LOG >

t = 2.0
evals = sp.linalg.eigh(H_.toarray(), b=S_.toarray())[0]
# evals = sp.linalg.eigh(H_.toarray())[0]
true_trace = np.exp(-1j * evals * t).sum()

# ------------------------------------------------------------------ stochastic trace
tlog("=== 5. Stochastic Chebyshev trace ===")                    ### LOG >

H_normalized, t_normalized = normalize(H_, t)
trace_est = random_trace_evaluation(
    H_normalized, t=t_normalized, S=100, tau=1e-14, overlap_matrix=S_, verbose=True  # <-- verbose!
)
# H_normalized, t_normalized = normalize(H_, t)
# trace_est = random_trace_evaluation(
#     H_normalized, t=t_normalized, S=10000, tau=1e-14, overlap_matrix=None, verbose=True  # <-- verbose!
# )

# ------------------------------------------------------------------ summary
tlog("=== 6. Results ===")                                       ### LOG >
print(f"Exact trace      : {true_trace.real:.6f} {true_trace.imag:+.6f}j")
print(f"Estimated trace  : {trace_est.real:.6f} {trace_est.imag:+.6f}j")
print(f"Relative error   : {abs(trace_est-true_trace)/abs(true_trace):.2e}")
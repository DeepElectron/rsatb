#!/usr/bin/env python3
"""
Trace estimator for e^{-itA} using the textbook Chebyshev expansion

    e^{-izx} = J0(z) + 2 Σ_{m=1}^∞ (-i)^m J_m(z) T_m(x)

*No* spectral scaling is carried out here – the caller must provide
A whose eigenvalues already lie in [-1, 1].
"""

import numpy as np
from scipy import sparse
from scipy.linalg import eigvalsh
from scipy.special import jv
import graphblas as gb
from graphblas import monoid, binary
from itertools import count
from graphblas import unary
import sksparse.cholmod as cholmod
io = gb.io
        
def random_trace_evaluation(A_csr, t, sampling_number, tau, rng = None, overlap_matrix = None, verbose = False):
    if verbose:
        print("─" * 46)
        print("  Random-trace estimator for  e^{-i t A}")
        print("─" * 46)

        if np.isscalar(t) or (isinstance(t, np.ndarray) and t.ndim == 0):
            print(f"  ▸ time t                : {t:.6g}")
        else:
            t = np.asarray(t, dtype=float)
            print(f"  ▸ times t (array)       : len={len(t)}, "
                f"[min={t.min():.4g}, max={t.max():.4g}]")

        print(f"  ▸ random vectors S      : {sampling_number}")
        print(f"  ▸ tolerance τ           : {tau:.1e}")

        if overlap_matrix is None:
            ov_txt = "None"
        else:
            ov_txt = f"{overlap_matrix.shape[0]}×{overlap_matrix.shape[1]}"
        print(f"  ▸ overlap matrix        : {ov_txt}")

        m, n = A_csr.shape
        print(f"  ▸ matrix A (CSR)        : {m}×{n}  "
            f"(nnz = {A_csr.nnz:,})")

        print("─" * 46)

    if overlap_matrix is not None:
        overlap_matrix = overlap_matrix.tocsc()
        overlap_matrix = cholmod.cholesky(overlap_matrix)
        
    if rng is None:
        rng = np.random.default_rng(1)
    
    if isinstance(t, float):
        t = np.array([t])
    A = gb.io.from_scipy_sparse(A_csr.astype(np.complex128, copy=False))
    n = A.nrows

    t_max = t[np.argmax(np.abs(t))]
    if verbose:
        print(f"worst-case time   t_max = {t_max:.4g}")
    steps_max = []

    # 2) Random test vectors (dense NumPy → GraphBLAS once)
    X0_np = (rng.standard_normal((n, sampling_number)) 
            + 1j*rng.standard_normal((n, sampling_number))) / np.sqrt(2)
    X0    = gb.Matrix.from_dense(X0_np, missing_value=0)
    X0conj = X0.apply(unary.conj)
    
    X0X0 = X0conj.ewise_mult(X0, op=binary.times).reduce_rowwise(monoid.plus).to_dense(0)
    steps_max.append(X0X0)
    X1  = -1j*A.mxm(X0)   

    if overlap_matrix is not None:
        
        X1 = X1.to_dense(0)              
        X1 = overlap_matrix.solve_A(X1)
        X1 = gb.Matrix.from_dense(X1, missing_value=0)

    X0X1 = X0conj.ewise_mult(X1, op=binary.times).new().reduce_rowwise(monoid.plus).to_dense(0)
    steps_max.append(X0X1)

    X_nm1, X_n = X0, X1
    
    for mp1 in count(2, 1):
        J_next = jv(mp1, t_max)
        coeff  = 2 * J_next

        if overlap_matrix is not None:
            X_mp1 = (A.mxm(X_n) * (-2j)).new().to_dense(0)
            X_mp1 = overlap_matrix.solve_A(X_mp1)
            X_mp1 = gb.Matrix.from_dense(X_mp1, missing_value=0).ewise_add(X_nm1).new()
        else:
            X_mp1 = (A.mxm(X_n) * (-2j)).ewise_add(X_nm1).new()

        X_mp1_coeff = (X_mp1 * coeff).new()                   

        norm = X_mp1_coeff.apply(unary.conj).ewise_mult(X_mp1_coeff, op=binary.times).new()
        norm = norm.reduce_rowwise(monoid.plus).to_dense(0)
        norm = np.sqrt(norm)
        norm = norm.max()

        if verbose:
            print(f"m = {mp1:2d} (norm = {norm:.2e})", flush=True)

        if norm < tau:
            max_term = mp1-1
            if verbose:
                print(f"converged at m = {mp1:2d} (norm = {norm:.2e})")
            break
        X0X_mp1 = X0conj.ewise_mult(X_mp1, op=binary.times).new().reduce_rowwise(monoid.plus).to_dense(0)
        steps_max.append(X0X_mp1)
        X_nm1, X_n = X_n, X_mp1

    steps_max = np.array(steps_max)/sampling_number

    J = jv(np.arange(max_term+1)[:, np.newaxis], t[np.newaxis,:])
    J[1:, :] *= 2.0

    traces = np.sum(J.T @ steps_max, axis=1)

    if len(t) == 1:
        return traces[0]
    else:       
        return traces
        

def make_symmetric(n, density, rng, t):
    """Make a real symmetric CSR matrix with spectrum scaled to [-1,1]."""
    M = sparse.random(n, n, density, format="csr", random_state=rng, dtype=np.complex128)
    A = (M + M.getH()) * 0.5
    # exact spectral radius
    norm_A = np.max(np.abs(eigvalsh(A.toarray())))
    t_tilde = t * norm_A
    return A / norm_A, t_tilde 

def normalize(A, t):
    """Make a real symmetric CSR matrix with spectrum scaled to [-1,1]."""

    norm_A = np.max(np.abs(eigvalsh(A.toarray())))
    t_tilde = t * norm_A
    return A / norm_A, t_tilde 
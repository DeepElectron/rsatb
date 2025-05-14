#!/usr/bin/env python3
"""
Trace estimator for e^{-itA} using the textbook Chebyshev expansion

    e^{-izx} = J0(z) + 2 Σ_{m=1}^∞ (-i)^m J_m(z) T_m(x)

*No* spectral scaling is carried out here - the caller must provide
A whose eigenvalues already lie in [-1, 1].
"""

import numpy as np
from scipy import sparse
from scipy.linalg import eigvalsh
from scipy.special import jv
import scipy as sp
from typing import Union
import graphblas as gb
from graphblas import monoid, binary
from itertools import count
from graphblas import unary
import sksparse.cholmod as cholmod
from ..utils.ozaki import ozaki_residues

io = gb.io

def _tgf(
        A_csr: sp.sparse.csr_array, 
        t_max: Union[int, float], 
        B_csr: Union[sp.sparse.csr_array, None]=None, 
        v0: np.ndarray=None, 
        tol: float=1e-14, 
        verbose=False,
        trace: bool=True,
        ): # generate the chebyshev decomposition vector for e^{-itA}
    
    n, S = v0.shape
    assert A_csr.shape[0] == A_csr.shape[1] == n, "The matrix and the samples's dimension should matched!"

    # normalize v0
    v0 = v0 / np.linalg.norm(v0, axis=0, keepdims=True)

    if verbose:
        print("─" * 46)
        print("  Random-trace estimator for  e^{-i t A}")
        print("─" * 46)

        print(f"  ▸ random vectors S      : {S}")
        print(f"  ▸ tolerance τ           : {tol:.1e}")

        if B_csr is None:
            ov_txt = "None"
        else:
            ov_txt = f"{B_csr.shape[0]}x{B_csr.shape[1]}"
        print(f"  ▸ overlap matrix        : {ov_txt}")

        m, n = A_csr.shape
        print(f"  ▸ matrix A (CSR)        : {m}x{n}  "
            f"(nnz = {A_csr.nnz:,})")

        print("─" * 46)

    if B_csr is not None:
        B_solve = cholmod.cholesky(B_csr.tocsc())

    ## The graph blas implementation
    A = gb.io.from_scipy_sparse(A_csr.astype(np.complex128, copy=False))
    steps_max = []

    # 2) Random test vectors (dense NumPy → GraphBLAS once)
    X0    = gb.Matrix.from_dense(v0, missing_value=0) # [n, S]
    X0conj = X0.apply(unary.conj) # [n, S]
    
    if trace:
        Xi = X0conj.ewise_mult(X0, op=binary.times).reduce_rowwise(monoid.plus).to_dense(0) # [S]
    else:
        Xi = X0 # [n, S]
    steps_max.append(Xi)
    X1  = -1j*A.mxm(X0)  # [n, S]  

    if B_csr is not None:
        X1 = X1.to_dense(0)
        X1 = B_solve.solve_A(X1)
        X1 = gb.Matrix.from_dense(X1, missing_value=0) # [n, S]

    if trace:
        Xi = X0conj.ewise_mult(X1, op=binary.times).new().reduce_rowwise(monoid.plus).to_dense(0) # [n, S]
    else:
        Xi = X1 # [S]

    steps_max.append(Xi)

    for mp1 in count(2, 1):
        J_next = jv(mp1, t_max)
        coeff  = 2 * J_next

        if B_csr is not None:
            Xi = (A.mxm(X1) * (-2j)).new().to_dense(0)
            Xi = B_solve.solve_A(Xi)
            Xi = gb.Matrix.from_dense(Xi, missing_value=0).ewise_add(X0).new()
        else:
            Xi = (A.mxm(X1) * (-2j)).ewise_add(X0).new()              

        norm = Xi.apply(unary.conj).ewise_mult(Xi, op=binary.times).new() # [n, S]
        norm = norm.reduce_rowwise(monoid.plus).to_dense(0) # [S]
        norm = np.sqrt(norm) * np.abs(coeff)
        norm = norm.max()

        if verbose:
            print(f"m = {mp1:2d} (norm = {norm:.2e})", flush=True)

        if norm < tol:
            max_term = mp1-1
            if verbose:
                print(f"converged at m = {mp1:2d} (norm = {norm:.2e})")
            break
        
        X0, X1 = X1, Xi
        if trace:
            Xi = X0conj.ewise_mult(Xi, op=binary.times).new().reduce_rowwise(monoid.plus).to_dense(0)
        steps_max.append(Xi)
    
    steps_max = n * np.array(steps_max) / S # [n_cheb, n, S] / [n_cheb, S]

    return steps_max


def trace_tgf(
        steps_max,
        t
):
    if np.isscalar(t) or (isinstance(t, np.ndarray) and t.ndim == 0):
        t = np.array([t])

    max_term = len(steps_max)
    J = jv(np.arange(max_term)[:, np.newaxis], t[np.newaxis,:])
    J[1:, :] *= 2.0

    traces = np.sum(J.T @ steps_max, axis=1) # [t, n_cheb] x [n_cheb, S] -> [t, S] -> [t]

    return traces
    

def mat_tgf(
    steps_max,
    t,
    v0,
):
    max_term = len(steps_max)
    J = jv(np.arange(max_term)[:, np.newaxis], t[np.newaxis,:])
    J[1:, :] *= 2.0
        
    vecs = np.einsum("ji,jkl -> ikl", J, steps_max)  # [t, n_cheb] x [n_cheb, n, S] -> [t, n, S]

    # need careful thought about how to avoid memory explode here.

    pass    


def nocc(
    A_csr: sp.sparse.csr_array, 
    B_csr: Union[sp.sparse.csr_array, None]=None, 
    v0: np.ndarray=None,
    mu0: Union[float, int]=0.,
    tol: float=1e-14, 
    verbose=False,
    Mcut=1000,
    R=1e5,
    kBT=0.025,
    ntol: float=1e-4,
    method="pole"
):
    assert method == "pole", "Currently, only pole method is supported!"

    poles, res = ozaki_residues(M_cut=1000)

    poles = poles * kBT

    # scale the A_csr

    # compute the chebyshev power vectors
    steps_max = _tgf(
                A_csr=A_csr,
                t_max=4 * np.log(10) / poles[0],
                B_csr=B_csr,
                v0=v0,
                tol=tol,
                verbose=verbose,
                trace=True
                )

    # compute the docc in each spectrum
    tmin = 0.
    ts = np.array([])
    cs = np.array([])
    for ip, p in poles[::-1]:
        tmax = 4 * np.log(10) / p

        # generating integration grid
        if tmax > tmin:
            h = np.sqrt(12*ntol / (tmax*(p**2)))
            new_ts = np.append(ts, np.linspace(tmin, tmax, num=int((tmax-tmin)/h)))
            cs = np.concatenate(cs, trace_tgf(steps_max=steps_max, t=new_ts))
            ts = np.concatenate(ts, new_ts)

        # compute the expectation at given time steps
        
        np.trapezoid(y=(1/2*np.pi) * cs*np.exp(-p*ts), x=ts)

        if tmax > tmin:
            tmin = tmax






    




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

def normalize_small_matrix(A, t):
    """Make a real symmetric CSR matrix with spectrum scaled to [-1,1]."""

    norm_A = np.max(np.abs(eigvalsh(A.toarray())))
    t_tilde = t * norm_A
    return A / norm_A, t_tilde 

def normalize_large_matrix(A, t, B=None):
    """Make a real symmetric CSR matrix with spectrum scaled to [-1,1]."""

    
    max_eig = sp.sparse.linalg.eigsh(A, k=1, M=B, which="LA", return_eigenvectors=False)
    min_eig = sp.sparse.linalg.eigsh(A, k=1, M=B, which="SA", return_eigenvectors=False)
    mid_eig = (max_eig + min_eig) / 2
    norm_A = (max_eig[0] - min_eig[0])/2

    if B is None:
        A = A - mid_eig[0] * sp.sparse.eye(A.shape[0])
    else:
        A = A - mid_eig[0] * B

    return A / norm_A, t * norm_A, mid_eig[0]
#!/usr/bin/env python3
import os


# pick your target number of cores
nthreads = 8

# OpenMP (SuiteSparse:GraphBLAS, numba, etc.)
os.environ["OMP_NUM_THREADS"]        = str(nthreads)

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
import matplotlib.pyplot as plt
import graphblas as gb
from graphblas import monoid, binary
from itertools import count
from graphblas import unary
import sksparse.cholmod as cholmod
io = gb.io
        
def random_trace_evaluation(A_csr, t, S, rng, tau, overlap_matrix = None):

    if overlap_matrix is not None:
        overlap_matrix = overlap_matrix.tocsc()
        overlap_matrix = cholmod.cholesky(overlap_matrix)
        

    
    if isinstance(t, float):
        t = np.array([t])
    A = gb.io.from_scipy_sparse(A_csr.astype(np.complex128, copy=False))
    n = A.nrows

    t_max = t[np.argmax(np.abs(t))]
    steps_max = []

    # 2) Random test vectors (dense NumPy → GraphBLAS once)
    X0_np = (rng.standard_normal((n, S)) 
            + 1j*rng.standard_normal((n, S))) / np.sqrt(2)
    X0    = gb.Matrix.from_dense(X0_np, missing_value=0)
    X0conj = X0.apply(unary.conj)
    
    X0X0 = X0conj.ewise_mult(X0, op=binary.times).reduce_rowwise(monoid.plus).to_dense(0)
    steps_max.append(X0X0)
    J0, J1 = jv(0, t_max), jv(1, t_max)
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

        X_mp1 = (A.mxm(X_n) * (-2j)).ewise_add(X_nm1).new()

        if overlap_matrix is not None:
            X_mp1 = X_mp1.to_dense(0)              
            X_mp1 = overlap_matrix.solve_A(X_mp1)
            X_mp1 = gb.Matrix.from_dense(X_mp1, missing_value=0)

        X_mp1_coeff = (X_mp1 * coeff).new()                   

        norm = X_mp1_coeff.apply(unary.conj).ewise_mult(X_mp1_coeff, op=binary.times).new()
        norm = norm.reduce_rowwise(monoid.plus).to_dense(0)
        norm = np.sqrt(norm)
        norm = norm.max()
        if norm < tau:
            max_term = mp1-1
            break
        X0X_mp1 = X0conj.ewise_mult(X_mp1, op=binary.times).new().reduce_rowwise(monoid.plus).to_dense(0)
        steps_max.append(X0X_mp1)
        X_nm1, X_n = X_n, X_mp1

    steps_max = np.array(steps_max)/S

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
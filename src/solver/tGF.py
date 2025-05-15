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
from sparse_dot_mkl import dot_product_mkl
from itertools import count
import sksparse.cholmod as cholmod
from ..utils.ozaki import ozaki_residues
from scipy.fft import ifft, fftshift
from .time_GF_gb import normalize_large_matrix

def _evolve_mkl(
        A_csr: sp.sparse.csr_array, 
        dt: Union[int, float], 
        v0: np.ndarray,
        vt0: Union[np.ndarray, None],
        B_csr: Union[sp.sparse.csr_array, None]=None, 
        tol: float=1e-14, 
        verbose=False,
        ): # generate the chebyshev decomposition vector for e^{-itA}
    """
        computing the moments of chebyshev approximation times the random state starting at t0.
        <v0|e^{-iAt}|vt0>, or e^{-iAt}|vt0> if trace if false.

        A_csr: The Hamiltonian
        t_max: The largest time step in the approximated domain.
        v0: The original random state
        vt0: The random state evolved at time step t0, by default, it is equals to v0.
        B_csr: The overlap matrix.
    """
    
    n, S = v0.shape
    assert A_csr.shape[0] == A_csr.shape[1] == n, "The matrix and the samples's dimension should matched!"

    # normalize v0
    v0 = v0 / np.linalg.norm(v0, axis=0, keepdims=True)
    if vt0 is None:
        vt0 = v0
    if abs(dt) < tol:
        return vt0

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
    Xt = 0.

    # 2) Random test vectors (dense NumPy → GraphBLAS once)
    Xt0    = vt0 # [n, S]
    Xt += jv(0, dt) * vt0
    
    Xt1  = -1j*dot_product_mkl(A_csr, Xt0)  # [n, S]  

    if B_csr is not None:
        Xt1 = B_solve.solve_A(Xt1)

    Xt += jv(1, dt) * 2 * Xt1

    for mp1 in count(2, 1):
        J_next = jv(mp1, dt)
        coeff  = 2 * J_next


        if B_csr is not None:
            Xtmp = -2j * dot_product_mkl(A_csr, Xt1)
            Xtmp = B_solve.solve_A(Xtmp)
            Xtmp = Xtmp + Xt0
        else:
            Xtmp = -2j * dot_product_mkl(A_csr, Xt1) + Xt0

        norm = np.linalg.norm(Xtmp, axis=0) # [n, S]
        norm = norm * np.abs(coeff)
        norm = norm.max()

        if verbose:
            print(f"m = {mp1:2d} (norm = {norm:.2e})", flush=True)

        if norm < tol:
            if verbose:
                print(f"converged at m = {mp1:2d} (norm = {norm:.2e})")
            break
        
        Xt0, Xt1 = Xt1, Xtmp
    
        Xt += coeff * Xtmp
    
    # vals, vecs = np.linalg.eigh(A_csr.todense())
    # # print(np.exp(vecs.shape))
    # op = (vecs @ np.diag(np.exp(-1j*vals*dt))) @ vecs.conj().T
    # lbl = np.ascontiguousarray(op) @ vt0
    # print(Xt-lbl, (Xt* Xt.conj()).sum(), (lbl.conj()*lbl).sum())

    return Xt

def _evolve_gbl(
        A_csr: sp.sparse.csr_array, 
        dt: Union[int, float], 
        v0: np.ndarray,
        vt0: Union[np.ndarray, None],
        B_csr: Union[sp.sparse.csr_array, None]=None, 
        tol: float=1e-14, 
        verbose=False,
        ): # generate the chebyshev decomposition vector for e^{-itA}
    """
        computing the moments of chebyshev approximation times the random state starting at t0.
        <v0|e^{-iAt}|vt0>, or e^{-iAt}|vt0> if trace if false.

        A_csr: The Hamiltonian
        t_max: The largest time step in the approximated domain.
        v0: The original random state
        vt0: The random state evolved at time step t0, by default, it is equals to v0.
        B_csr: The overlap matrix.
    """
    
    n, S = v0.shape
    assert A_csr.shape[0] == A_csr.shape[1] == n, "The matrix and the samples's dimension should matched!"

    # normalize v0
    v0 = v0 / np.linalg.norm(v0, axis=0, keepdims=True)
    if vt0 is None:
        vt0 = v0
    if abs(dt) < tol:
        return vt0

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
    Xt = 0.

    # 2) Random test vectors (dense NumPy → GraphBLAS once)
    Xt0    = gb.Matrix.from_dense(vt0, missing_value=0) # [n, S]
    Xt += jv(0, dt) * vt0
    
    Xt1  = -1j*A.mxm(Xt0)  # [n, S]  

    if B_csr is not None:
        Xt1 = Xt1.to_dense(0)
        Xt1 = B_solve.solve_A(Xt1)
        Xt1 = gb.Matrix.from_dense(Xt1, missing_value=0) # [n, S]

    Xt += jv(1, dt) * 2 * Xt1.to_dense(0)

    for mp1 in count(2, 1):
        J_next = jv(mp1, dt)
        coeff  = 2 * J_next


        if B_csr is not None:
            Xtmp = (A.mxm(Xt1) * (-2j)).new().to_dense(0)
            Xtmp = B_solve.solve_A(Xtmp)
            Xtmp = gb.Matrix.from_dense(Xtmp, missing_value=0).ewise_add(Xt0).new()
        else:
            Xtmp = (A.mxm(Xt1) * (-2j)).ewise_add(Xt0).new()     

        norm = Xtmp.apply(unary.conj).ewise_mult(Xtmp, op=binary.times).new() # [n, S]
        norm = norm.reduce_rowwise(monoid.plus).to_dense(0).real # [S]
        norm = np.sqrt(norm) * np.abs(coeff)
        norm = norm.max()

        if verbose:
            print(f"m = {mp1:2d} (norm = {norm:.2e})", flush=True)

        if norm < tol:
            if verbose:
                print(f"converged at m = {mp1:2d} (norm = {norm:.2e})")
            break
        
        Xt0, Xt1 = Xt1, Xtmp
    
        Xt += coeff * Xtmp.to_dense(0)

    return Xt


def trace_tgf(
        A_csr: sp.sparse.csr_array,
        t: Union[list, np.ndarray, int, float],
        v0: np.ndarray,
        B_csr: Union[sp.sparse.csr_array, None]=None,
        tol: float=1e-14, 
        verbose=False,
):
    # If dt is None, we evolve the system in one shot, and generate the value for different time step during postprocessing
    # If dt is given, we perform a step-by-step time evolution, each time evolve the system by dt, and computing the trace along side

    if np.isscalar(t) or (isinstance(t, np.ndarray) and t.ndim == 0):
        t = np.array([t])
    elif isinstance(t, list):
        t = np.array(t)
    else:
        assert isinstance(t, np.ndarray)

    An, norm_A, Emid, Emax, Emin = normalize_large_matrix(A_csr, B_csr)
    t = t * norm_A
    v0 = v0 / np.linalg.norm(v0, axis=0, keepdims=True)
    v0conj = v0.conj()
    n = A_csr.shape[0]

    ids = np.argsort(t)
    tsorted = t[ids]
    ptid = ids[tsorted>0]
    pt = t[ptid]
    ntid = ids[tsorted<=0]
    nt = t[ntid]

    vt0 = v0
    tt_ = 0.
    trp = []

    for it, tt in enumerate(pt):
        xt = _evolve_mkl(A_csr=An, B_csr=B_csr, dt=tt-tt_, v0=v0, vt0=vt0, tol=tol, verbose=verbose)
        traces = (xt * v0conj).mean(axis=-1).sum() * n # [n, S], [n, S] -> [S] -> [1]
        tt_ = tt
        trp.append(traces)
        vt0 = xt
    
    trp = np.array(trp)

    vt0 = v0
    tt_ = 0.
    trn = []

    for it, tt in enumerate(nt[::-1]):
        xt = _tgf(A_csr=An, B_csr=B_csr, dt=tt-tt_, v0=v0, vt0=vt0, verbose=verbose)
        traces = (xt * v0conj).mean(axis=-1).sum() * n # [n, S], [n, S] -> [n] -> [1]
        tt_ = tt
        trn.append(traces)
        vt0 = xt

    trn = np.array(trn)

    tr = np.zeros_like(t).astype(A_csr)
    tr[ptid] = trp
    tr[ntid] = trn[::-1]
    tr = tr / np.exp(1j*Emid*(t/norm_A))

    return tr

def efermi(
    A_csr: sp.sparse.csr_array, 
    B_csr: Union[sp.sparse.csr_array, None]=None, 
    v0: np.ndarray=None,
    nocc: Union[int, float]=0.,
    mu0: Union[float, int, None]=None,
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
    _, norm_A, Emid, Emax, Emin = normalize_large_matrix(A_csr, B_csr)

    if mu0 is None:
        mu0 = Emid

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

        if tmax > tmin:
            tmin = tmax
    
    
    # add the largest term
    N = (4*kBT*res * np.trapezoid(y=-(cs[None,:]*np.exp(1j*mu0*ts[None,:])*np.exp(-poles[:,None]*ts[None,:])).imag, x=ts, axis=-1)).sum()
    N += -R * np.trapezoid(y=-(cs*np.exp(1j*mu0*ts[None,:])*np.exp(-R*ts)).imag, x=ts)

    if abs(N - nocc) > ntol:
        if N > nocc:
            Emax = mu0
        else:
            Emin = mu0
        mu0 = (Emax+Emin) / 2
    else:
        return mu0
    
    # continue if not converge
    while True:
        N = (4*kBT*res * np.trapezoid(y=-(cs[None,:]*np.exp(1j*mu0*ts[None,:])*np.exp(-poles[:,None]*ts[None,:])).imag, x=ts, axis=-1)).sum()
        N += -R * np.trapezoid(y=-(cs*np.exp(1j*mu0*ts)*np.exp(-R*ts)).imag, x=ts)

        if abs(N - nocc) > ntol:
            if N > nocc:
                Emax = mu0
            else:
                Emin = mu0
            mu0 = (Emax+Emin) / 2
        else:
            break

    return mu0
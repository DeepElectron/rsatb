#!/usr/bin/env python3
"""
Solve S X = B with PETSc using either
1. sparse Cholesky      (fastest if it fits in memory), or
2. block Conjugate Gradient + AMG preconditioner (scales to 10⁶+ dof).

Requires: petsc (≥3.20), petsc4py, numpy, scipy.
"""
import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc
from typing import Union
import src.linalg.hpddm as hpddm
from .hpddm import MatrixCSR
import ctypes
import time
from .petscop import csr_to_sbaij, dense_to_petsc
from ..utils.tools import float2complex, complex2float, real_csr_to_complex, real_dense_to_complex

_cached = {}
def _get_ksp(Ap, m_block, rtol, maxit):
    key = (Ap.getComm().tompi4py().py2f(), Ap.getSize()[0], m_block)
    if key in _cached:                     # reuse matrices of the same size
        ksp, Xp = _cached[key]
        ksp.setTolerances(rtol=rtol, max_it=maxit)
        return ksp, Xp
    
    ksp = PETSc.KSP().create(comm=Ap.getComm())
    ksp.setOperators(Ap)
    ksp.setType('hpddm')
    ksp.setHPDDMType(PETSc.KSP.HPDDMType.BCG)
    # ksp.setTolerances(rtol=1e-6, max_it=500)
    pc = ksp.getPC()
    pc.setBDDCLevels(1)
    # pc.setLevelsSubKSPType(PETSc.KSP.Type.PREONLY)
    # pc.setLevelsSubPCType(PETSc.PC.Type.ILU)
    pc.setType('hpddm')
    pc.setHPDDMCoarseCorrectionType(pc.HPDDMCoarseCorrectionType.ADDITIVE)
    

    # ksp.setReusePreconditioner(True)
    pc.setReusePreconditioner(True)
    opts = PETSc.Options()

    opts.insertString(
        # "-pc_hpddm_levels_1_eps_nev 6 "
        # "-pc_hpddm_levels_1_sub_pc_type ilu "
        # "-pc_hpddm_overlap 1 "
        "-pc_hpddm_coarse_pc_type hypre "
        "-pc_hpddm_coarse_pc_hypre_type boomeramg "

        # "-pc_hpddm_coarse_pc_factor_mat_solver_type pastix"
    )
    
    ksp.setFromOptions()
    ksp.setUp()
    Xp = PETSc.Mat().createDense(size=(Ap.getSize()[0], m_block),
                                 comm=Ap.getComm())
    _cached[key] = (ksp, Xp)
    return ksp, Xp

def bsolve(A: Union[sp.csr_matrix, PETSc.Mat], B: Union[np.ndarray, PETSc.Mat], rtol: float=1e-8, maxit: int=500):
    # switch the datatype
    if isinstance(A, sp.csr_array):
        Asize = A.shape
    else:
        Asize = A.getSize()

    if isinstance(B, np.ndarray):
        if B.ndim == 1:
            B = B[:,None]
        Bsize = B.shape
    else:
        Bsize = B.getSize()

    assert Asize[0] == Asize[1] == Bsize[0]

    if isinstance(A, sp.csr_array):
        if A.dtype in [np.float32, np.float64, np.float128]:
            A = float2complex(A)
            A_is_float = True
        else:
            A_is_float = False
        A = csr_to_sbaij(A)
    else:
        A_is_float = False

    if isinstance(B, np.ndarray):
        if B.dtype in [np.float32, np.float64, np.float128]:
            B = real_dense_to_complex(B)
            B_is_float = True
        else:
            B_is_float = False
        
        B = dense_to_petsc(B)
    else:
        B_is_float = False

    Xp = bsolve_petsc(Ap=A, Bp=B, rtol=rtol, maxit=maxit)

    X = Xp.getDenseArray(readonly=True).copy(order="F") # important to copy, otherwise the pointer would be freed since ksp is garbage collected

    if A_is_float and B_is_float:
        X = complex2float(X)

    return X

def bsolve_petsc(Ap: PETSc.Mat, Bp: PETSc.Mat, rtol: float=1e-8, maxit: int=500):
    S = Bp.getSize()[1]
    M = Ap.getSize()[0]
    # print("A hermitian loss", Aseq)
    ksp, Xp = _get_ksp(Ap, S, rtol, maxit)
    ksp.matSolve(Bp, Xp) # single call does block CG
    # Xp is still PETsc mat format
    return Xp

def bsolve_hpddm(
        A: Union[sp.csr_array, ctypes.POINTER(hpddm.MatrixCSR)], 
        B: np.ndarray,
        X: np.ndarray=None,
        rtol=1e-6,
        maxiter=100,
        M=None,
        # specific tuning param
        spd=False,
        method="bgmres",
        msg=""
        ):
    
    if isinstance(A, sp.csr_matrix):
        # transform it into hpddm format
        n, m = A.shape
        indptr, indices, data = A.indptr, A.indices, A.data
        if data.dtype != hpddm.scalar:
            data = data.astype(hpddm.scalar, copy=False)

        nnz = len(data)
        A_ = hpddm.matrixCSRCreate(n,m,nnz,data,indptr,indices,False)
        destroy = True
    else:
        A_ = A
        destroy = False
    
    # check if B is F contiguous:
    if not B.flags.f_contiguous:
        B = np.asfortranarray(B)
    
    if B.dtype != hpddm.scalar:
        B = B.astype(hpddm.scalar, copy=False)
        
    if X is None:
        X = np.zeros_like(B)
    else:
        if X.dtype != hpddm.scalar:
            X = X.astype(hpddm.scalar, copy=False)
    
    @hpddm.precondFunc
    def precond(y, x, n, m):
        # pass
        if hpddm.scalar is not hpddm.underlying: # underlining is c_double, so complex number took two connected address space?
            factor = 2
        else:
            factor = 1
        ptr = ctypes.cast(x, ctypes.POINTER(hpddm.underlying * factor * m * n))
        x = np.frombuffer(ptr.contents)
        ptr = ctypes.cast(y, ctypes.POINTER(hpddm.underlying * factor * m * n))
        # x,y input are two array pointer on the first element, same as C.
        y = np.frombuffer(ptr.contents)
        x = x.view(dtype = hpddm.scalar)
        y = y.view(dtype = hpddm.scalar)
        if m > 1:
            x = np.reshape(x, (n, m), order = 'F')
            y = np.reshape(y, (n, m), order = 'F')
        # x[:] = lu.solve(y[:])
        if M is not None:
            x[:] = M @ y
        else:
            x[:] = y
    
    opt = hpddm.optionGet()
    cmd  = "-hpddm_krylov_method "+method+" -hpddm_max_it "+str(maxiter) + " -hpddm_tol "+str(rtol) + " "
    if spd:
        cmd += "-hpddm_operator_spd 1 "
    
    cmd = cmd + msg
    cmd = cmd.encode('ascii')
    
    hpddm.optionParse(opt, cmd)
    hpddm.solve(A_, precond, B, X)

    if destroy:
        hpddm.matrixCSRDestroy(ctypes.byref(A_))
    
    return X
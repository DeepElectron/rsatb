import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc
from typing import Union
import time
from .petscop import csr_to_sbaij, dense_to_petsc, csr_to_aij
from ..utils.tools import float2complex, complex2float, real_csr_to_complex, real_dense_to_complex

def matmulpetsc(A: Union[sp.csr_matrix, np.ndarray, PETSc.Mat], B: Union[sp.csr_matrix, np.ndarray, PETSc.Mat], return_Mat=False):
    start = time.time()
    if isinstance(A, sp.csr_array) or isinstance(A, np.ndarray):
        Asize = A.shape
    elif isinstance(A, PETSc.Mat):
        Asize = A.getSize()
    else:
        raise TypeError("The matmul here only support mat type [sp.csr_matrix, np.ndarray, PETSc.Mat].")

    if isinstance(B, np.ndarray) or isinstance(B, sp.csr_matrix):
        if B.ndim == 1:
            B = B[:,None]
        Bsize = B.shape
    elif isinstance(B, PETSc.Mat):
        Bsize = B.getSize()
    else:
        raise TypeError("The matmul here only support mat type [sp.csr_matrix, np.ndarray, PETSc.Mat].")
    
    Csize = (Asize[0], Bsize[1])

    assert Asize[1] == Bsize[0], "The matrix for matmul of A and B should have shape [N,M], [M,K]"

    if isinstance(A, sp.csr_array) or isinstance(A, np.ndarray):
        if A.dtype in [np.float32, np.float64, np.float128]:
            A = float2complex(A)
            A_is_float = True
        else:
            A_is_float = False
        if isinstance(A, sp.csr_array):
            A = csr_to_aij(A)
        elif isinstance(A, np.ndarray):
            A = dense_to_petsc(A)
    else:
        A_is_float = False

    if isinstance(B, np.ndarray) or isinstance(B, sp.csr_array):
        if B.dtype in [np.float32, np.float64, np.float128]:
            B = real_dense_to_complex(B)
            B_is_float = True
        else:
            B_is_float = False

        if isinstance(B, sp.csr_array):
            B = csr_to_aij(B)
        elif isinstance(B, np.ndarray):
            B = dense_to_petsc(B)
    else:
        B_is_float = False

    Xp = PETSc.Mat().createDense(size=Csize, comm=A.getComm())

    print("MVP prepare time: ", time.time() - start)
    start = time.time()
    A.matMult(B, result=Xp)

    print("MVP real time: ", time.time() - start)

    if not return_Mat:
        Xp = Xp.getDenseArray(readonly=True).copy(order="F")
        
        if A_is_float and B_is_float:
            X = complex2float(X)
    
    return Xp



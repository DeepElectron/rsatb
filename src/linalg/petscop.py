from petsc4py import PETSc
from petsc4py.PETSc import Vec, Mat
import numpy as np
import scipy.sparse as sp


# two question:
# 1. how to transform and transform it back of np scipy format data and PETSc type data
# 2. what will be the case of MPI situation.
def dense_to_petsc(B: np.ndarray) -> PETSc.Mat:
    """Wrap a NumPy (n x m) array as a PETSc dense matrix (no copy)."""
    n, m = B.shape
    return PETSc.Mat().createDense(size=(n, m), array=B,
                                   comm=PETSc.COMM_WORLD)

def csr_to_sbaij(A_csr: sp.csr_matrix) -> PETSc.Mat:
    """Clone a SciPy hermitian CSR into a PETSc SBAIJ matrix."""
    comm = PETSc.COMM_WORLD
    m = A_csr.shape[0]
    Istart, Iend = PETSc.Mat().create(comm=comm).getOwnershipRange(m) if comm.size > 1 else (0, m)
    A_loc = A_csr[Istart:Iend]
    indptr, indices, data = A_loc.indptr, A_loc.indices, A_loc.data
    row_ids = np.repeat(np.arange(Istart, Iend), np.diff(indptr))
    lower_mask = indices >= row_ids  # diag + strictly upper

    indices_lt = indices[lower_mask].astype(np.int32, copy=False)
    data_lt    = data[lower_mask]
    counts = np.bincount(row_ids[lower_mask] - Istart, minlength=Iend - Istart)
    indptr_lt = np.empty_like(indptr, dtype=np.int32)
    indptr_lt[0] = 0
    indptr_lt[1:] = np.cumsum(counts).astype(np.int32)

    assert indptr_lt[-1] == len(indices_lt) == len(data_lt)

    A_p = PETSc.Mat().createSBAIJ(
            size=A_csr.shape,
            bsize=1,
            csr=(indptr_lt, indices_lt, data_lt),
            comm=PETSc.COMM_WORLD
            )
    
    # print(A_p.getType())
    A_p.setOption(PETSc.Mat.Option.HERMITIAN, True)
    A_p.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    # A_p.setOption(PETSc.Mat.Option.SPD, True)
    A_p.assemble()
    # print(A_p.isSymmetric(), A_p.isHermitian())
    # print(A_p.getType())
    return A_p

def csr_to_aij(A_csr: sp.csr_matrix) -> PETSc.Mat:
    """Clone a SciPy hermitian CSR into a PETSc SBAIJ matrix."""
    comm = PETSc.COMM_WORLD
    m = A_csr.shape[0]
    Istart, Iend = PETSc.Mat().create(comm=comm).getOwnershipRange(m) if comm.size > 1 else (0, m)
    A_loc = A_csr[Istart:Iend]
    indptr, indices, data = A_loc.indptr, A_loc.indices, A_loc.data

    A_p = PETSc.Mat().createAIJ(
            size=A_csr.shape,
            csr=(indptr, indices, data),
            comm=PETSc.COMM_WORLD
            )
    
    # print(A_p.getType())
    # A_p.setOption(PETSc.Mat.Option.HERMITIAN, True)
    # A_p.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    # A_p.setOption(PETSc.Mat.Option.SPD, True)
    A_p.assemble()
    # print(A_p.isSymmetric(), A_p.isHermitian())
    # print(A_p.getType())
    return A_p



def T(A):
    AT = A.duplicate(True)
    AT.transpose()

    return AT

def Tconj(A):
    ATconj = A.duplicate(True)
    ATconj.hermitianTranspose()

    return ATconj

def rowsum(A): # sum over index 1
    return A.getRowSum()

def colsum(A): # sum over index 2
    return T(A).getRowSum()

def getsub(A, rows, cols, todense=False):
    isrow = PETSc.IS().createGeneral(rows, comm=PETSc.COMM_WORLD)
    iscol = PETSc.IS().createGeneral(cols, comm=PETSc.COMM_WORLD)

    B = A.getSubMatrix(isrow, iscol)

    if todense:
        B = B.convert('dense')

    return B


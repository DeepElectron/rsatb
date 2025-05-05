from petsc4py import PETSc
import numpy as np
import scipy.sparse as sp
from src.linalg.bsolve import bsolve
from src.linalg.petscop import csr_to_sbaij
import scipy.linalg as sla

if __name__ == "__main__":
    n = 3000
    eye = sp.csr_array(np.eye(n)) + 0j
    asp = sp.random(n, n, density=0.01, format="csr")
    asp = asp + sp.random(n, n, density=0.01, format="csr") * 1j
    asp = asp + asp.conj().T
    eigvals, _ = sp.linalg.eigsh(asp, which="SA", k=6)
    asp = asp - eye * (eigvals[0] - 0.8)
    adense = asp.todense()

    comm = PETSc.COMM_WORLD
    m = asp.shape[0]
    Istart, Iend = PETSc.Mat().create(comm=comm).getOwnershipRange(m) if comm.size > 1 else (0, m)
    A_loc = asp[Istart:Iend]
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
            size=asp.shape,
            bsize=1,
            csr=(indptr_lt, indices_lt, data_lt),
            comm=PETSc.COMM_WORLD
            )
    
    # print(A_p.getType())
    A_p.setOption(PETSc.Mat.Option.HERMITIAN, True)
    A_p.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    # A_p.setOption(PETSc.Mat.Option.SPD, True)
    A_p.assemble()

    # A_p = csr_to_sbaij(asp)
    # Apdense = A_p.convert("dense").getDenseArray()

    # print(np.abs(Apdense-Apdense.conj().T).max(), np.abs(Apdense - adense).max())

    for i in range(10):
        B = np.random.randn(n,1) + 0j
        ans = sp.linalg.spsolve(asp, B[:,0])[:,None]

        ans_ = bsolve(
            A=A_p, B=B
        )

        # print(ans_, ans)
        print(np.abs(ans - ans_).max())


# """
# Direct Hermitian‑SPD solver with PaStiX (Cholesky) through petsc4py.

# Run with:  mpiexec -n 4 python solve_pastix_chol.py -mat_pastix_thread 8
# """

# from petsc4py import PETSc
# import numpy as np

# def build_hermitian_spd(n: int, comm=PETSc.COMM_WORLD) -> PETSc.Mat:
#     """
#     Build a random complex Hermitian positive‑definite matrix in SBAIJ format.
#     Only the lower triangle is stored (PaStiX needs that for Cholesky).
#     """
#     A = PETSc.Mat().create(comm=comm)
#     A.setSizes([n, n])
#     A.setType(PETSc.Mat.Type.SBAIJ)          # symmetric storage, block size = 1
#     A.setFromOptions()
#     A.setUp()

#     rng = np.random.default_rng(42)
#     Istart, Iend = A.getOwnershipRange()

#     for i in range(Istart, Iend):
#         # diagonal (real & ≥ 0)
#         diag = 2.0 + rng.random()
#         A[i, i] = diag

#         # strictly lower part
#         for j in range(0, i):
#             r = rng.standard_normal()
#             im = rng.standard_normal()
#             val = r + 1j * im
#             A[i, j] = val  # only one triangle needed
#             A[j, i] = r - 1j * im

#     A.assemblyBegin()
#     A.assemblyEnd()

#     # Flag for the solver: Hermitian SPD
#     A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
#     A.setOption(PETSc.Mat.Option.HERMITIAN, True)

#     # a = A.convert("dense").getDenseArray()
#     # print(np.abs(a-a.conj().T).max())

#     return A


# def solve_with_pastix(A: PETSc.Mat, B: PETSc.Mat | PETSc.Vec):
#     """
#     Factorises A with PaStiX (Cholesky) and solves A X = B.
#     Accepts a dense matrix B (multi‑RHS) or a PETSc vector.
#     """
#     ksp = PETSc.KSP().create(comm=A.comm)
#     ksp.setOperators(A)
#     ksp.setType(PETSc.KSP.Type.PREONLY)

#     pc = ksp.getPC()
#     pc.setType(PETSc.PC.Type.CHOLESKY)        # -> PCCHOLESKY
#     pc.setFactorSolverType("pastix")          # backend

#     # optional: PaStiX tuning through PETSc options
#     opts = PETSc.Options()
#     # Number of PaStiX threads: may also be provided on the CLI
#     opts["mat_pastix_thread"] = 8
#     # memory limit (bytes) or verbose level etc. if you want:
#     # opts["mat_pastix_maxmem"] = 8_000_000_000
#     # opts["mat_pastix_verbose"] = 1

#     ksp.setFromOptions()
#     ksp.setUp()

#     # Solve -------------------------------------------------------------
#     if isinstance(B, PETSc.Mat):             # many RHS: use MatMatSolve
#         X = B.duplicate()
#         ksp.matSolve(B, X)
#     else:                                    # single RHS vector
#         X = B.duplicate()
#         ksp.solve(B, X)

#     # Residual check
#     r = B.duplicate()
#     A.matMult(X, r)          # r = A·X
#     r.axpy(-1.0, B)       # r = r - B
#     res = r.norm()
#     if A.comm.rank == 0:
#         print(f"‖A X – B‖₂  = {res:.3e}")

#     return X


# if __name__ == "__main__":
#     n     = PETSc.Options().getInt("n", 10)   # override with -n <size>
#     nrhs  = PETSc.Options().getInt("nrhs", 3)   # -nrhs <# RHS>

#     A = build_hermitian_spd(n)

#     # --- RHS: dense matrix with <nrhs> columns --------------------------
#     B = PETSc.Mat().createDense([n, nrhs], comm=A.comm)
#     B.setRandom()

#     X = solve_with_pastix(A, B)

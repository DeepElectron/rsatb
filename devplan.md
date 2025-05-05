# obj
A POC demo to compute DOS with Linear Scale of sparse H, S
- The data format design
    - atom sorting algorithm
    - sparse matrix structure
    - block hyrachircal sparse matrix structure
- Krylov subspace method
    - Parralize across k point
- PCG method
    - As fast as possible
    - Block PCG support
    - a dynamic allocated PCG: Matrix size: M, sample number: S
        M large, S medium/small: BPCG
        M medium, S large: BPCG/Direct Solver
        M medium, S small/medium: BPCG
        M small, S large/medium/small: Direct Solver
- random GF
    - Parralize across samples S

# implementation plan:
The algorithm would looks like:
Generating a lot of random samples -> perform batch Krylov algorithm (which involves BPCG for S^{-1}|v>) -> use the returning matrix to compute quantities in a batch of E points.
1. Batch Krylov algorithm:
    - PCG: 
    - support MPI
    - preconditioner acceleration: https://arxiv.org/pdf/1603.08970
    - two choice:
        - 1. Simplest one, just batched MINRES/CG use shift invariance subspace of S^{-1}H, need stablized lanczos algorithm
        - 2. rational krylov method, when the first does not converge, or the dimension of krylov space grows too large.


# useful libraries
1. DBCSR: https://cp2k.github.io/dbcsr/develop/
2. pyamg
3. PETsc: https://petsc.org/release/
4. STRUMPACK: https://portal.nersc.gov/project/sparse/strumpack//

Below are the Python‑first libraries that let you spin up a *proof‑of‑concept* inverse‑overlap solve in minutes but still scale to ‘‘serious’’ sizes later.  I list them in the rough order I reach for them during prototyping, with one‑line install hints and what each piece buys you.

| level‑of‑effort | package (pip/conda) | what you get out‑of‑the‑box | why it is handy for an LCAO **S** | quick install |
|-----------------|--------------------|-----------------------------|----------------------------------|---------------|
| **zero extra deps** | **SciPy** `scipy.sparse.linalg` | CG / MINRES, incomplete‑LU (`spilu`), LinearOperator wrappers | baseline Krylov driver you can swap preconditioners into; 100 k–200 k AO is still fine on a laptop | `pip install scipy` citeturn1search0 |
| **single‑wheel AMG** | **PyAMG** | Smoothed‑aggregation & classical algebraic multigrid; `.aspreconditioner()` returns a SciPy‑compatible object | vanishes condition‑number problems in 5–10 iterations; perfect for rapid tests on randomly‑ordered sparse **S** | `pip install pyamg` citeturn2view0 |
| **sparse Cholesky** | **scikit‑sparse** (`sksparse.cholmod`) | CHOLMOD super‑nodal \(LL^\top\); rank‑k *update/downdate* built in | drop‑in substitute for a dense Cholesky in orthonormalisation loops; symbolic factor reused across SCF steps | `pip install scikit-sparse` (needs SuiteSparse headers) citeturn4view0 |
| **turn‑key HPC** | **petsc4py** | Full PETSc stack: ICC, ASM, GAMG AMG, nested‑dissection reordering, plus GPU back‑ends | lets the same prototype run on 1 CPU core or 10 000 GPUs without code changes; great for benchmarking solver choices | `pip install mpi4py petsc petsc4py` (pre‑built wheels as of PETSc 3.23) citeturn3view0 |
| **hierarchical low‑rank** | **STRUMPACK‑python / PySTRUMPACK** | HSS/HODLR approximate Cholesky/LU; MPI + CUDA ready | one command builds a linear‑scaling direct preconditioner for million‑orbital γ‑point supercells | `pip install strumpack-python` (needs OpenBLAS ≥0.3.25) citeturn0search3turn0search8 |
| **pure‑Python \( \mathcal H^2 \)** | **h2tools** | educational H‑/H²‑matrix builders & MVPs | useful sandbox for experimenting with block‑low‑rank compressions before wiring in a C++ backend | `pip install h2tools` citeturn6view0 |

### Minimal working snippets
```python
# 1) baseline CG + AMG
import scipy.sparse.linalg as spla, pyamg, scipy.sparse as sp, numpy as np
S = sp.load_npz("overlap.npz")          # CSR/BSR matrix
b = np.random.randn(S.shape[0])
ml = pyamg.smoothed_aggregation_solver(S)   # setup once
M  = ml.aspreconditioner()                  # LinearOperator
x, info = spla.cg(S, b, M=M, tol=1e-8)
```

```python
# 2) one‑line sparse Cholesky solve (scikit‑sparse)
from sksparse.cholmod import cholesky
factor = cholesky(S)   # symbolic+numeric; reuse across SCF iterations
x = factor(b)          # apply S⁻¹b in ~O(N log N) after reordering
```

```python
# 3) PETSc drop‑in (petsc4py)
from petsc4py import PETSc
A = PETSc.Mat().createAIJ(size=S.shape, csr=(S.indptr, S.indices, S.data))
ksp = PETSc.KSP().create()
ksp.setOperators(A); ksp.setType('cg'); ksp.getPC().setType('gamg')
ksp.setTolerances(rtol=1e-8); ksp.setFromOptions()
x = A.getVecLeft();  bvec = A.createVecRight();  bvec.setArray(b)
ksp.solve(bvec, x)
```

### Which one should I start with?

1. **Need a five‑minute demo?** SciPy + PyAMG (#1) is 100 % Python wheels—no compilation headaches.
2. **Want to check scaling >10⁵ AO on a workstation?** Add scikit‑sparse (#2) for a real factorisation benchmark.
3. **Planning HPC production later?** Prototype the workflow in petsc4py (#4); switch PC type (`icc`, `gamg`, etc.) from the command line (`-pc_type gamg`) without touching your Python.

All six libraries speak NumPy/SciPy sparse formats, so you can swap them in or mix them (e.g. PyAMG preconditioner inside SciPy CG, or PETSc KSP calling a STRUMPACK preconditioner) with only a few lines of glue.

That should be everything you need to get a fast POC running—happy hacking!
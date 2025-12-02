import sys
print("SYS.PATH:", sys.path)
try:
    import rsatb
    print("RSATB FILE:", rsatb.__file__)
except ImportError:
    print("RSATB not found")

from rsatb.solver import RSATB
import numpy as np
from ase.io import read
from rsatb.data import AtomicData
import pytest
import os

def test_orth_vs_dual():
    # Setup common parameters
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data/diamond")
    
    basis = {"C":"2s2p1d"}
    atoms = read(os.path.join(data_dir, "diamond_sc111.vasp"))
    adata = AtomicData.from_ase(atoms, r_max=7.5)
    
    # Run 1: Dual Basis (Standard)
    print("Running Dual Basis Calculation...")
    rsatb_dual = RSATB(basis=basis, atomicdata=adata, n_group=1, enable_mpi=True, verbose=True)
    rsatb_dual.setup(
        n=1,
        rsmethod="gauss",
        kpoints=np.array([0.,0.,0.]),
        HFile=os.path.join(data_dir, "H_diamond_sc111.h5"),
        orbroot=data_dir + "/", # orbroot expects trailing slash or directory
        orblist=["C_gga_7au_100Ry_2s2p1d.orb"],
        tol=1e-4,
        sptol=1e-4,
        orthogonalize=False
    )
    out_dual = rsatb_dual.compute_dos(dE=0.5)
    dos_dual = out_dual[1] # omegas, dos
    
    # Run 2: Orthogonalization
    print("Running Orthogonalization Calculation...")
    rsatb_orth = RSATB(basis=basis, atomicdata=adata, n_group=1, enable_mpi=True, verbose=True)
    rsatb_orth.setup(
        n=1,
        rsmethod="gauss",
        kpoints=np.array([0.,0.,0.]),
        HFile=os.path.join(data_dir, "H_diamond_sc111.h5"),
        orbroot=data_dir + "/",
        orblist=["C_gga_7au_100Ry_2s2p1d.orb"],
        tol=1e-4,
        sptol=1e-4,
        orthogonalize=True
    )
    out_orth = rsatb_orth.compute_dos(dE=0.5)
    dos_orth = out_orth[1]
    
    # Compare
    # Note: Since we are using random vectors (n=1), the results might differ slightly due to different random seeds 
    # if the seed is not fixed or if the operations affect the random stream differently.
    # However, for a large enough system or enough samples, they should converge.
    # But here we are using the SAME random seed if possible, or we should check if we can fix it.
    # RSATB uses PETSc random.
    # Let's check if the results are reasonably close.
    # Actually, if we use the same random vector, the results should be IDENTICAL (machine precision) 
    # because mathematically \tilde{H} = S^{-1/2} H S^{-1/2} and we are solving (z - \tilde{H}) x = ... 
    # vs (z S - H) x = ... -> (z - S^{-1} H) x = ... 
    # Wait, dual basis solves generalized eigenvalue problem or uses S-inner product?
    # KPM uses Chebyshev moments.
    # Dual basis: moments of S^{-1} H.
    # Orthogonal: moments of S^{-1/2} H S^{-1/2}.
    # These operators have the same spectrum.
    # The DOS trace should be the same.
    # Trace(f(S^{-1}H)) vs Trace(f(S^{-1/2} H S^{-1/2})).
    # Trace(S^{-1/2} H S^{-1/2}) = Trace(H S^{-1}). Yes.
    
    # However, stochastic trace estimation depends on the random vectors.
    # If we use the same random vector in the standard basis, say v.
    # In dual basis, we compute v^T f(S^{-1}H) v.
    # In orthogonal basis, we work in orthogonalized basis. The vector should be transformed?
    # No, usually we generate random vectors in the working basis.
    # If we generate v in orthogonal basis, it corresponds to S^{-1/2} v in original basis?
    # This might introduce differences if we just compare "run with random v" vs "run with random v'".
    # But if we assume the code handles "setup_sample" similarly (generating random vector of size N),
    # then we are comparing Trace(A) estimated by v^T A v vs Trace(B) estimated by v^T B v.
    # Since Trace(A) = Trace(B), the expectations are the same.
    # But individual realizations might differ if A and B act differently on v.
    
    # Let's check the difference.
    diff = np.abs(dos_dual - dos_orth)
    max_diff = np.max(diff)
    print(f"Max difference in DOS: {max_diff}")
    
    # We might need a loose tolerance if random seeds are different or effective vectors are different.
    # But let's try.
    
    # To be more rigorous, we could try to fix the seed if exposed.
    # But for now, let's see the magnitude.
    
    assert max_diff < 1e-2, f"DOS difference too large: {max_diff}"

if __name__ == "__main__":
    test_orth_vs_dual()

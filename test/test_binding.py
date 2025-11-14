from rsatb.core import Solver
from rsatb.data import AtomicData, OrbitalMapper, _keys
import ase.io as io
import pytest
import numpy as np
import mpi4py

N = "4"*3
idp = OrbitalMapper(basis={"C":"2s2p1d"})
atomicdata = AtomicData.from_ase(io.read("./data/diamond/diamond_sc"+N+".vasp"), r_max=7.5)
# atomicdata_ex = AtomicData.from_ase(io.read("./data/diamond/diamond_sc"+N+".vasp"), r_max=15.0)
# atomicdata = AtomicData.from_ase(io.read("./data/moire/test1.vasp"), r_max=7.5)
# atomicdata_ex = AtomicData.from_ase(io.read("./data/moire/test1.vasp"), r_max=7.5)
atomicdata = idp(atomicdata)
HFile = "./data/diamond/H_diamond_sc"+N+".h5"
SFile = "./data/diamond/S_diamond_sc"+N+".h5"
# HFile = "/nessa/users/zhanghao/dev/rsatb/test/data/diamond/C2.0/hamiltonians.h5"
# SFile = "/nessa/users/zhanghao/dev/rsatb/test/data/diamond/C2.0/overlaps.h5"

solver = Solver(
        atomicdata[_keys.ATOM_TYPE_KEY].flatten(), 
        atomicdata[_keys.CELL_KEY],
        atomicdata[_keys.POSITIONS_KEY],
        atomicdata[_keys.EDGE_INDEX_KEY].T.copy(order="C"), # copy is needed, since .T is a view, view does not count as c++ param
        idp.atom_norb,
        atomicdata[_keys.EDGE_CELL_SHIFT_KEY],
        mpi4py.MPI.COMM_WORLD,
        "-info stacking.log:ksp,pc,eps -log_view :profiling.log"
) # c++ implementation
# -pc_type asm -pc_asm_overlap 0 -pc_asm_blocks 1 -pc_asm_sub_mat_type seqaijmkl -sub_pc_type icc -sub_pc_factor_levels 1 -mat_ordering_type rcm
# "-prefix_push pc_hpddm_ -prefix_push levels_1_ -pc_type asm -eps_nev 40 -sub_pc_type cholesky -sub_pc_factor_mat_solver_type mumps -st_pc_factor_mat_solver_type mumps -prefix_pop -prefix_push coarse_ -p 24 -pc_factor_mat_type matseqaijmkl -prefix_pop -define_subdomains -has_neumann -prefix_pop"

solver.setup_graph()
# solver.setup_auxgraph(
#     atomicdata_ex[_keys.EDGE_INDEX_KEY].T.copy(order="C"),
#     atomicdata_ex[_keys.EDGE_CELL_SHIFT_KEY],
# )

solver.setup_sample(1, "gauss")

solver.setup_K(0.,0.,0.)

# solver.setup_S_h5(SFile)

file_orb0 = ["C_gga_7au_100Ry_2s2p1d.orb"]
root = "./data/diamond/"
solver.setup_S_orb(file_orb0, root)

solver.Ssqrtm(1e-3,1e-3, 20, 0.31)

# solver.test_Sinv()

# solver.export_S("/nessa/users/zhanghao/dev/rsatb/test/data/diamond")
# Ax=b -> (ZAZ)(Z^{-1}x)=ZAx=Zb,
# print("Here?")
# solver.setup_H_h5(HFile)

# solver.setup_R_orb(file_orb0, root)

# nt = solver.get_nt(0.025)

# ac, omegas = solver.compute_ac(100, 5000, 2000.0, 16.0797722399)

# print(ac)

# print(omegas)

# TODO: Test Fermi-Dirac, compute trace to see whether the fermi-dirac operator is correct or not.
# Compute Fermi energy
# Benchmark AC's results.
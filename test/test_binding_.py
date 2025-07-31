from rsatb.core import Solver
from rsatb.data import AtomicData, OrbitalMapper, _keys
import ase.io as io
import pytest
import numpy as np
import mpi4py

if __name__== "__main__":
    idp = OrbitalMapper(basis={"Al":"4s4p1d"})
    atomicdata = AtomicData.from_ase(io.read("./test/data/Al101010.vasp"), r_max=7.5)
    atomicdata = idp(atomicdata)
    HFile = "./test/data/H_Al101010.h5"
    SFile = "./test/data/S_Al101010.h5"
    
    solver = Solver(
            HFile, SFile,
            atomicdata[_keys.ATOM_TYPE_KEY].flatten(), 
            atomicdata[_keys.POSITIONS_KEY],
            atomicdata[_keys.EDGE_INDEX_KEY].T.copy(order="C"), # copy is needed, since .T is a view, view does not count as c++ param
            idp.atom_norb,
            atomicdata[_keys.EDGE_CELL_SHIFT_KEY],
            mpi4py.MPI.COMM_WORLD
    ) # c++ implementation

    solver.setup_graph()

    solver.setup_HS(np.array([0.,0.,0.]))

    solver.setup_sample(1, "gauss")

    nt = solver.get_nt(0.025)

    print(nt)

    dos = solver.compute_dos(100, "none", np.array([0,0,0]))
    print(dos)
    
    import matplotlib.pyplot as plt

    plt.plot(dos[0])
    plt.savefig("./dos.png", dpi=100)
    plt.close()

    # to run: OMP_NUM_THREADS=20 python ./test/test_binding_.py
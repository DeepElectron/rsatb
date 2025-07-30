from rsatb.core import Solver
from rsatb.data import AtomicData, OrbitalMapper, _keys
import ase.io as io
import pytest
import numpy as np
import mpi4py

if __name__== "__main__":
    idp = OrbitalMapper(basis={"Al":"4s4p1d"})
    atomicdata = AtomicData.from_ase(io.read("./test/data/Al.vasp"), r_max=7.5)
    atomicdata = idp(atomicdata)
    HFile = "./test/data/H_Al.h5"
    SFile = "./test/data/S_Al.h5"
    
    solver = Solver(
            HFile, SFile,
            atomicdata[_keys.ATOM_TYPE_KEY].flatten(), 
            atomicdata[_keys.POSITIONS_KEY],
            atomicdata[_keys.EDGE_INDEX_KEY].T.copy(order="C"), # copy is needed, since .T is a view
            idp.atom_norb,
            atomicdata[_keys.EDGE_CELL_SHIFT_KEY],
            mpi4py.MPI.COMM_WORLD
    ) # c++ implementation

    solver.setup_graph()

    solver.setup_HS(np.array([0.,0.,0.]))

        # print("Here, the setup_graph pass!")

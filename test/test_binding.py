from rsatb.rsatb.core import Solver
from rsatb.rsatb.data import AtomicData, OrbitalMapper, _keys
import ase.io as io
import pytest
import mpi4py

class TestBinding:
    idp = OrbitalMapper(basis={"Al":"4s4p1d"})
    atomicdata = AtomicData.from_ase(io.read("./test/data/Al.vasp"), r_max=7.5)
    atomicdata = idp(atomicdata)
    HFile = "./test/data/H_Al.h5"
    SFile = "./test/data/S_Al.h5"
    
    def test_init(self):
        solver = Solver(
                self.HFile, self.SFile,
                self.atomicdata[_keys.ATOM_TYPE_KEY].flatten(), 
                self.atomicdata[_keys.POSITIONS_KEY],
                self.atomicdata[_keys.EDGE_INDEX_KEY].T,
                self.idp.atom_norb,
                self.atomicdata[_keys.EDGE_CELL_SHIFT_KEY],
                mpi4py.MPI.COMM_WORLD
        ) # c++ implementation

        solver.setup_graph()

        # print("Here, the setup_graph pass!")

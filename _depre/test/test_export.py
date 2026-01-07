import numpy as np
from rsatb.solver import RSATB
from rsatb.data import AtomicData, _keys
import ase.io
import os

# Load test data
root = "./test/data/silicon/"
pos = ase.io.read(os.path.join(root, "POSCAR"), format="vasp")
ad = AtomicData.from_ase(pos, r_max=2.5)

# print(ad.edge_index, "\n", ad.edge_cell_shift)
# Initialize Solver
solver = RSATB(basis={"Si": "2s2p1d"}, atomicdata=ad, verbose=True, enable_mpi=True)

# Setup
kpoints = np.array([[0.0, 0.0, 0.0]])
HFile = os.path.join(root, "Si2.0/hamiltonians.h5")
orbroot = root
orblist = ["Si_gga_7au_100Ry_2s2p1d.orb"]

solver.setup(n=1, rsmethod="gauss", kpoints=kpoints, HFile=HFile, orblist=orblist, orbroot=orbroot, orthogonalize=False, explicit_orth=False)

solver.solver.setup_K(*solver.__sub_kpoints__[0])
solver.solver.setup_R_orb(solver.orblist, solver.orbroot)

# solver.export_H("/nessa/users/zhanghao/dev/rsatb/test/data/silicon/exported/")
solver.export_S("/nessa/users/zhanghao/dev/rsatb/test/data/silicon/exported/")
solver.export_Rx("/nessa/users/zhanghao/dev/rsatb/test/data/silicon/exported/")
solver.export_Ry("/nessa/users/zhanghao/dev/rsatb/test/data/silicon/exported/")
solver.export_Rz("/nessa/users/zhanghao/dev/rsatb/test/data/silicon/exported/")
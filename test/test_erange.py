import numpy as np
from rsatb.solver import RSATB
from rsatb.data import AtomicData, _keys
from rsatb.utils.make_kpoints import kmesh_sampling, abacus_kpath
import ase.io
import os

root = "/nessa/users/zhanghao/dev/rsatb/test/data/diamond/"
pos = ase.io.read(os.path.join(root, "diamond_sc111.vasp"), format="vasp")
ad = AtomicData.from_ase(pos, r_max=7.5)

# Initialize Solver
solver = RSATB(basis={"C": "2s2p1d"}, n_group=4, atoms=ad, r_max=None, verbose=True, enable_mpi=True)

# Setup
kpoints, kdist, hsp = abacus_kpath(
    pos,
    kpath=[
        [0.5,0.5,0.5,10],
        [0.,0.,0.,10],
        [0.5,0.,0.5,1]
    ]
)

HFile = os.path.join(root, "H_diamond_sc111.h5")
orbroot = root
orblist = ["C_gga_7au_100Ry_2s2p1d.orb"]
psdlist = ["C_ONCV_PBE-1.0.upf"]

solver.setup(n=1, rsmethod="gauss", kpoints=kpoints, HFile=HFile, orblist=orblist, orbroot=orbroot, psdlist=psdlist, psdroot=orbroot, orthogonalize=True, explicit_orth=False)

# We can pass RxFile, RyFile, RzFile during setup or rely on setup_R_orb if using orbitals.
# Since we used orblist in setup, setup_R_orb will be called internally by compute -> _setup_kpoint.

# Define tasks

results = solver.compute_energy_range()

if __name__ == "__main__":
    # print(results)
    print(np.max(results), np.min(results))
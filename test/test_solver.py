from rsatb.solver import RSATB
import numpy as np
from ase.io import read
from rsatb.data import AtomicData

atoms = read("/nessa/users/zhanghao/dev/rsatb/test/data/diamond/diamond_sc888.vasp")
adata = AtomicData.from_ase(atoms, r_max=7.5)
rsatb = RSATB(basis={"C":"2s2p1d"}, atomicdata=adata, n_group=1, enable_mpi=True, verbose=True)

rsatb.setup(
    n=1,
    rsmethod="gauss",
    kpoints=np.array([0.5,0.5,0.5]),
    HFile="/nessa/users/zhanghao/dev/rsatb/test/data/diamond/H_diamond_sc888.h5",
    orbroot="/nessa/users/zhanghao/dev/rsatb/test/data/diamond/",
    orblist=["C_gga_7au_100Ry_2s2p1d.orb"],
    tol=1e-6,
    sptol=1e-4
)

# rsatb.setup_S_2c(rsatb.orblist, rsatb.orbroot, rsatb.tol, rsatb.sptol, rsatb.maxit)
# rsatb.solver.test_Sinv()
out = rsatb.compute_dos(dE=1.0)

# print(out)


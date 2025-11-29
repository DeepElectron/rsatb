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
    kpoints=np.array([0.,0.,0.]),
    HFile="/nessa/users/zhanghao/dev/rsatb/test/data/diamond/H_diamond_sc888.h5",
    orbroot="/nessa/users/zhanghao/dev/rsatb/test/data/diamond/",
    orblist=["C_gga_7au_100Ry_2s2p1d.orb"],
    tol=1e-4,
    sptol=1e-4
)

# rsatb.setup_S_2c(rsatb.orblist, rsatb.orbroot, rsatb.tol, rsatb.sptol, rsatb.maxit)
# rsatb.solver.test_Sinv()
out = rsatb.compute_dos(dE=0.02)


if rsatb.rank == 0:
    arr = np.zeros((2,out[0].shape[1]))
    arr[0] = out[0][0]
    arr[1] = out[1][0]
    np.save("/nessa/users/zhanghao/dev/rsatb/test/data/diamond/diamond_dos.npy", arr)


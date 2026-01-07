from rsatb.solver import RSATB
import numpy as np
from ase.io import read
from rsatb.data import AtomicData

atoms = read("test/data/diamond/diamond_sc111.vasp")
adata = AtomicData.from_ase(atoms, r_max=7.5)
rsatb = RSATB(basis={"C":"2s2p1d"}, atomicdata=adata, n_group=1, enable_mpi=True, verbose=True)

rsatb.setup(
    n=1,
    rsmethod="gauss",
    kpoints=np.array([0.,0.,0.]),
    HFile="test/data/diamond/H_diamond_sc111.h5",
    orbroot="test/data/diamond/",
    orblist=["C_gga_7au_100Ry_2s2p1d.orb"],
    tol=1e-5,
    sptol=1e-5,
    explicit_orth=False,
    orthogonalize=False,
)

# rsatb.setup_S_2c(rsatb.orblist, rsatb.orbroot, rsatb.tol, rsatb.sptol, rsatb.maxit)
# rsatb.solver.test_Sinv()
out = rsatb.compute_dos(dE=5)

print(out[1])

# Estimated Emax : (164.892,0)  Emin : (-8.11778,0)
# Estimated Emax : (164.892,0)  Emin : (-8.11778,0)


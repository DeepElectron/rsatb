from mpi4py import MPI
import rsatb
from rsatb import RSATB

kpoints = [
    [0. , 0. , 0. ]
    ]

rsatb = RSATB.from_file(npar=1, filename="./data/diamond/diamond_sc666.vasp", r_max_per_type=[3.7], type_norb=[13], verbose=True)

rsatb.setup(
    Hfile="", kpoints=kpoints, orb_names=["C_gga_7au_100Ry_2s2p1d.orb"], 
    orb_dirs="./data/diamond/", psd_names=["C_ONCV_PBE-1.0.upf"], psd_dirs="./data/diamond/", 
    tol=1e-7, sptol=1e-5, maxiter=0, method="graph",
    precond="S", nsamples=1)

rsatb.test_solve()

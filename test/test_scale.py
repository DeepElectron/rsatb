from mpi4py import MPI
import rsatb
from rsatb import RSATB

rank = MPI.COMM_WORLD.Get_rank()

kpoints = [
    [0. , 0. , 0. ]
    ]

rsatb = RSATB.from_file(npar=1, filename="./data/diamond/diamond_sc888.vasp", r_max_per_type=[3.7], type_norb=[13], verbose=True)

rsatb.setup(
    Hfile="./data/diamond/H_diamond_sc888.h5", kpoints=kpoints, orb_names=["C_gga_7au_100Ry_2s2p1d.orb"], 
    orb_dirs="./data/diamond/", psd_names=["C_ONCV_PBE-1.0.upf"], psd_dirs="./data/diamond/", 
    tol=1e-7, sptol=1e-5,
    precond="S", nsamples=1)

dos, omega = rsatb.compute_dos(1)

if rank == 0:
    import matplotlib.pyplot as plt
    plt.imshow(dos.reshape(len(kpoints), -1), aspect='auto', cmap="bwr", vmax=1, vmin=-1)
    plt.colorbar()
    plt.savefig("dos.png")

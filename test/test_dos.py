from mpi4py import MPI
import rsatb
from rsatb import RSATB

rank = MPI.COMM_WORLD.Get_rank()

kpoints = [
    [0.5 , 0.5 , 0.5 ],
    [0.45, 0.45, 0.45],
    [0.4 , 0.4 , 0.4 ],
    [0.35, 0.35, 0.35],
    [0.3 , 0.3 , 0.3 ],
    [0.25, 0.25, 0.25],
    [0.2 , 0.2 , 0.2 ],
    [0.15, 0.15, 0.15],
    [0.1 , 0.1 , 0.1 ],
    [0.05, 0.05, 0.05],
    [0.  , 0.  , 0.  ],
    [0.05, 0.  , 0.05],
    [0.1 , 0.  , 0.1 ],
    [0.15, 0.  , 0.15],
    [0.2 , 0.  , 0.2 ],
    [0.25, 0.  , 0.25],
    [0.3 , 0.  , 0.3 ],
    [0.35, 0.  , 0.35],
    [0.4 , 0.  , 0.4 ],
    [0.45, 0.  , 0.45],
    [0.5 , 0.  , 0.5 ]
    ]

rsatb = RSATB.from_file(npar=5, filename="./data/diamond/diamond_sc111.vasp", r_max_per_type=[3.7], type_norb=[13])

rsatb.setup(Hfile="./data/diamond/C2.0/hamiltonians.h5", kpoints=kpoints, orb_names=["C_gga_7au_100Ry_2s2p1d.orb"], orb_dirs="./data/diamond/", psd_names=["C_ONCV_PBE-1.0.upf"], psd_dirs="./data/diamond/", precond="H", nsamples=500)

dos, omega = rsatb.compute_dos(200)

if rank == 0:
    import matplotlib.pyplot as plt
    plt.imshow(dos.reshape(len(kpoints), -1), aspect='auto', cmap="bwr", vmax=1, vmin=-1)
    plt.colorbar()
    plt.savefig("dos.png")

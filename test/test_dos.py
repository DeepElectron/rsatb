from mpi4py import MPI
from rsatb import LCAOIntegralInput, LCAOModel, PreconditionerOptions, RSATB, SetupOptions

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

model_runner = RSATB.from_file(npar=5, filename="./data/diamond/diamond_sc111.vasp", r_max_per_type=[3.7], type_norb=[13], verbose=True)

model_runner.setup_lcao(
    LCAOModel.hdf5(
        "./data/diamond/C2.0/hamiltonians.h5",
        integrals=LCAOIntegralInput(
            orb_names=["C_gga_7au_100Ry_2s2p1d.orb"],
            orb_dir="./data/diamond/",
            psd_names=["C_ONCV_PBE-1.0.upf"],
            psd_dir="./data/diamond/",
        ),
    ),
    SetupOptions(
        kpoints=kpoints,
        nsamples=500,
        preconditioner=PreconditionerOptions(mode="H"),
    ),
)

dos, omega = model_runner.compute_dos(200)

if rank == 0:
    import matplotlib.pyplot as plt
    plt.imshow(dos.reshape(len(kpoints), -1), aspect='auto', cmap="bwr", vmax=1, vmin=-1)
    plt.colorbar()
    plt.savefig("dos.png")

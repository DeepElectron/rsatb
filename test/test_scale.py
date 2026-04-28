from mpi4py import MPI
from rsatb import LCAOIntegralInput, LCAOModel, PreconditionerOptions, RSATB, SetupOptions

rank = MPI.COMM_WORLD.Get_rank()

kpoints = [
    [0. , 0. , 0. ]
    ]

model_runner = RSATB.from_file(npar=1, filename="./data/diamond/diamond_sc888.vasp", r_max_per_type=[3.7], type_norb=[13], verbose=True)

model_runner.setup_lcao(
    LCAOModel.hdf5(
        "./data/diamond/H_diamond_sc888.h5",
        integrals=LCAOIntegralInput(
            orb_names=["C_gga_7au_100Ry_2s2p1d.orb"],
            orb_dir="./data/diamond/",
            psd_names=["C_ONCV_PBE-1.0.upf"],
            psd_dir="./data/diamond/",
        ),
    ),
    SetupOptions(
        kpoints=kpoints,
        nsamples=1,
        preconditioner=PreconditionerOptions(
            mode="H",
            tol=1e-7,
            sptol=1e-5,
        ),
    ),
)

ac_out, omega_out, properties = model_runner.compute_ac(1000, 1000, 300.0, 16.225300864, axis="abc")
properties["ac"] = ac_out
properties["omega"] = omega_out

if rank == 0:
    import numpy as np
    np.savez("./ac.npz", **properties)

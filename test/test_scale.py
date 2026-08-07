from mpi4py import MPI
from rsatb import (
    HamiltonianFile,
    LcaoModel,
    OrbitalFiles,
    PreconditionerOptions,
    PseudopotentialFiles,
    RSATB,
    SetupOptions,
    Structure,
)

rank = MPI.COMM_WORLD.Get_rank()

kpoints = [
    [0. , 0. , 0. ]
    ]

model = LcaoModel(
    hamiltonian=HamiltonianFile("./data/diamond/H_diamond_sc888.h5", format="deeptb"),
    orbitals=OrbitalFiles(["C_gga_7au_100Ry_2s2p1d.orb"], dir="./data/diamond/"),
    pseudopotentials=PseudopotentialFiles(["C_ONCV_PBE-1.0.upf"], dir="./data/diamond/"),
)

model_runner = RSATB(
    Structure.from_file("./data/diamond/diamond_sc888.vasp"),
    model,
    SetupOptions(
        kpoints=kpoints,
        nsamples=1,
        preconditioner=PreconditionerOptions(
            operator_form="sandwich",
            tol=1e-7,
            sptol=1e-5,
        ),
    ),
    npar=1,
)

ac_out, omega_out, properties = model_runner.compute_ac(1000, 1000, 300.0, 16.225300864, axis="abc")
properties["ac"] = ac_out
properties["omega"] = omega_out

if rank == 0:
    import numpy as np
    np.savez("./ac.npz", **properties)

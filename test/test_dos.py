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

structure = Structure.from_file("./data/diamond/diamond_sc111.vasp")

model = LcaoModel(
    hamiltonian=HamiltonianFile("./data/diamond/C2.0/hamiltonians.h5", format="deeptb"),
    orbitals=OrbitalFiles(["C_gga_7au_100Ry_2s2p1d.orb"], dir="./data/diamond/"),
    pseudopotentials=PseudopotentialFiles(["C_ONCV_PBE-1.0.upf"], dir="./data/diamond/"),
)

# The basis is read from the orbital files, not declared by the caller. These
# used to be passed in by hand as r_max_per_type=[3.7], type_norb=[13].
if rank == 0:
    print("type_symbols:", list(model.basis.type_symbols))
    print("type_norb:", list(model.basis.type_norb))
    print("cutoff_radius:", list(model.basis.cutoff_radius))
assert list(model.basis.type_norb) == [13]
assert abs(model.basis.cutoff_radius[0] - 3.7042) < 1e-3

model_runner = RSATB(
    structure,
    model,
    SetupOptions(
        kpoints=kpoints,
        nsamples=500,
        preconditioner=PreconditionerOptions(operator_form="sandwich"),
    ),
    npar=5,
)

dos, omega = model_runner.compute_dos(dE=1.4)

if rank == 0:
    import matplotlib.pyplot as plt
    plt.imshow(dos.reshape(len(kpoints), -1), aspect='auto', cmap="bwr", vmax=1, vmin=-1)
    plt.colorbar()
    plt.savefig("dos.png")

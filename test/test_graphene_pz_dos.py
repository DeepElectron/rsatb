from mpi4py import MPI
import numpy as np
import sys
from pathlib import Path

# Appended, not inserted: an installed rsatb must win. Inserting puts the
# source tree first, where the compiled _rsatb extension does not live unless
# the build happened in place.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from rsatb import (PreconditionerOptions, RSATB, SetupOptions, SkParameters, SktbModel,
                   sk_spline)

rank = MPI.COMM_WORLD.Get_rank()
if MPI.COMM_WORLD.Get_size() != 1:
    raise RuntimeError("test_graphene_pz_dos.py expects mpirun -np 1.")

DE_EV = 0.15  # graphene pz span ~17 eV -> ~115 Chebyshev steps
POINTS_PER_SEGMENT = 24

a_cc = 1.42
a = np.sqrt(3.0) * a_cc
c = 20.0
cell = [
    a, 0.0, 0.0,
    0.5 * a, 1.5 * a_cc, 0.0,
    0.0, 0.0, c,
]
positions = [
    0.0, 0.0, 0.5 * c,
    0.0, a_cc, 0.5 * c,
]


def interpolate_k_path(nodes, points_per_segment):
    kpoints = [nodes[0][1]]
    k_axis = [0.0]
    ticks = [0.0]

    for start, stop in zip(nodes[:-1], nodes[1:]):
        start = np.array(start[1], dtype=float)
        stop = np.array(stop[1], dtype=float)
        for step in range(1, points_per_segment + 1):
            kpoints.append((start + (stop - start) * step / points_per_segment).tolist())
            k_axis.append(float(len(kpoints) - 1))
        ticks.append(k_axis[-1])

    return kpoints, np.array(k_axis), ticks, [label for label, _ in nodes]


kpoints, k_axis, k_ticks, k_labels = interpolate_k_path(
    [
        (r"$\Gamma$", [0.0, 0.0, 0.0]),
        ("K", [2.0 / 3.0, 1.0 / 3.0, 0.0]),
        ("M", [0.5, 0.0, 0.0]),
        (r"$\Gamma$", [0.0, 0.0, 0.0]),
    ],
    POINTS_PER_SEGMENT,
)

graphene_pz = SktbModel(
    basis={"C": ["pz"]},
    hamiltonian=SkParameters(
        onsite={"C": {"pz": 0.0}},
        pairs={("C", "C"): {"pp_pi": sk_spline([0.0, 1.6], [-2.7, -2.7], 1.6)}},
    ),
)

edge_index = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
edge_shift_vec = [
    value
    for shift in [(0, 0, 0), (0, -1, 0), (1, -1, 0), (0, 0, 0), (0, 1, 0), (-1, 1, 0)]
    for value in shift
]

model_runner = RSATB.from_graph(
    atom_index=[0, 1],
    atom_type=[0, 0],
    edge_index=edge_index,
    edge_shift_vec=edge_shift_vec,
    cell=cell,
    pos=positions,
    model=graphene_pz,
    options=SetupOptions(
        kpoints=kpoints,
        nsamples=1,
        compute_bounds=False,
        energy_bounds=(-8.6, 8.6),
        preconditioner=PreconditionerOptions(target="none"),
    ),
    npar=1,
)

dos, omega = model_runner.compute_dos(dE=DE_EV)
del model_runner

if rank == 0:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    image = ax.imshow(
        dos.reshape(len(kpoints), -1).T,
        aspect="auto",
        origin="lower",
        extent=[k_axis[0], k_axis[-1], omega[0], omega[-1]],
        cmap="bwr",
        vmax=1e-1,
        vmin=-1e-1,
    )
    for tick in k_ticks:
        ax.axvline(tick, color="black", linewidth=0.7, alpha=0.35)
    ax.set_xticks(k_ticks)
    ax.set_xticklabels(k_labels)
    ax.set_xlim(k_axis[0], k_axis[-1])
    ax.set_xlabel("Graphene k path")
    ax.set_ylabel("Energy (eV)")
    fig.colorbar(image, ax=ax, label="k-resolved DOS")
    fig.tight_layout()
    fig.savefig("graphene_pz_kdos.png", dpi=200)

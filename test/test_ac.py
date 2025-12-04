import numpy as np
from rsatb.solver import RSATB
from rsatb.data import AtomicData, _keys
import ase.io
import os

# Load test data
root = "./test/data/diamond/"
pos = ase.io.read(os.path.join(root, "diamond_sc111.vasp"), format="vasp")
ad = AtomicData.from_ase(pos, r_max=7.5)

# Initialize Solver
solver = RSATB(basis={"C": "2s2p1d"}, atomicdata=ad, verbose=True, enable_mpi=True)

# Setup
kpoints = np.array([[0.0, 0.0, 0.0]])
HFile = os.path.join(root, "H_diamond_sc111.h5")
orbroot = root
orblist = ["C_gga_7au_100Ry_2s2p1d.orb"]

solver.setup(n=500, rsmethod="gauss", kpoints=kpoints, HFile=HFile, orblist=orblist, orbroot=orbroot, orthogonalize=False, explicit_orth=False)
# [     0.         -12897.22412389  -6527.89130329  -4096.92309184
#    -358.59765598]
# Compute AC using the optimized compute method
print("Computing AC Conductivity...")
# We can pass RxFile, RyFile, RzFile during setup or rely on setup_R_orb if using orbitals.
# Since we used orblist in setup, setup_R_orb will be called internally by compute -> _setup_kpoint.

# Define tasks
tasks = [
    {'task': 'ac', 'nt': 5, 'n_kpm': 10000, 'temp': 10000, 'mu': 0.0}
]

results = solver.compute(tasks)

if solver.rank == 0:
    omegas, ac = results[0]
    print("AC Calculation Complete.")
    print(f"Omegas shape: {omegas.shape}")
    print(f"AC shape: {ac.shape}")
    ac_reshaped = ac[0].reshape(-1, 6)
    print("First 5 AC values (xx component):", ac_reshaped[:5, 0])

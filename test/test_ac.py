import numpy as np
from rsatb.solver import RSATB
from rsatb.data import AtomicData, _keys
import ase.io
import os

root = "/nessa/users/zhanghao/dev/rsatb/test/data/diamond/"
pos = ase.io.read(os.path.join(root, "diamond_sc888.vasp"), format="vasp")
ad = AtomicData.from_ase(pos, r_max=7.5)

# Initialize Solver
solver = RSATB(basis={"C": "2s2p1d"}, atoms=ad, r_max=None, verbose=True, enable_mpi=True)

# Setup
kpoints = np.array([[0.0, 0.0, 0.0]])
HFile = os.path.join(root, "H_diamond_sc888.h5")
orbroot = root
orblist = ["C_gga_7au_100Ry_2s2p1d.orb"]
psdlist = ["C_ONCV_PBE-1.0.upf"]

solver.setup(n=1, rsmethod="gauss", kpoints=kpoints, HFile=HFile, orblist=orblist, orbroot=orbroot, psdlist=psdlist, psdroot=orbroot, orthogonalize=True, explicit_orth=False)

print("Computing AC Conductivity...")
# We can pass RxFile, RyFile, RzFile during setup or rely on setup_R_orb if using orbitals.
# Since we used orblist in setup, setup_R_orb will be called internally by compute -> _setup_kpoint.

# Define tasks
tasks = [
    {'task': 'ac', 'nt': 100, 'n_kpm': 10000, 'temp': 10000, 'mu': 16.225300864}
]

results = solver.compute(tasks)

if solver.rank == 0:
    import matplotlib.pyplot as plt
    omegas, ac = results[0]
    print(ac)
    print("AC Calculation Complete.")
    print(f"Omegas shape: {omegas.shape}")
    print(f"AC shape: {ac.shape}")
    ac_reshaped = ac[0].reshape(-1, 6)
    print("First 5 AC values (xx component):", ac_reshaped[:5, 0])

    plt.plot(omegas[0], ac_reshaped[:,0].real)
    plt.plot(omegas[0], ac_reshaped[:,0].imag)
    # plt.xlim(0,10)
    plt.savefig("./ac_0.png", dpi=100)
    plt.close()
    plt.plot(omegas[0], ac_reshaped[:,1].real)
    plt.plot(omegas[0], ac_reshaped[:,1].imag)
    # plt.xlim(0,10)
    plt.savefig("./ac_1.png", dpi=100)
    plt.close()
    plt.plot(omegas[0], ac_reshaped[:,2].real)
    plt.plot(omegas[0], ac_reshaped[:,2].imag)
    # plt.xlim(0,10)
    plt.savefig("./ac_2.png", dpi=100)
    plt.close()
    plt.plot(omegas[0], ac_reshaped[:,3].real)
    plt.plot(omegas[0], ac_reshaped[:,3].imag)
    # plt.xlim(0,10)
    plt.savefig("./ac_3.png", dpi=100)
    plt.close()
    plt.plot(omegas[0], ac_reshaped[:,4].real)
    plt.plot(omegas[0], ac_reshaped[:,4].imag)
    # plt.xlim(0,10)
    plt.savefig("./ac_4.png", dpi=100)
    plt.close()
    plt.plot(omegas[0], ac_reshaped[:,5].real)
    plt.plot(omegas[0], ac_reshaped[:,5].imag)
    # plt.xlim(0,10)
    plt.savefig("./ac_5.png", dpi=100)
    plt.close()

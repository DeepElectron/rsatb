"""Checks that rsatb read a Hamiltonian file right: its bands against ABACUS's own.

rsatb takes H(R) from the file and builds S from the orbital files; diagonalizing
the pair at the SCF k-points ABACUS used must give the BANDS_1.dat ABACUS printed.
Run it before computing anything from a new file -- a wrong atom order, orbital
file or spin setting shows up here as errors of eV, not 1e-4 eV.

usage: python check_bands.py al abacus | al nextham | gaas_soc nextham | gaas_soc abacus
"""
import pathlib
import sys

from mpi4py import MPI  # noqa: F401  -- MPI is initialized before rsatb uses it
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from rsatb import HamiltonianFile, LcaoModel, OrbitalFiles, RSATB, SetupOptions, Structure

HERE = pathlib.Path(__file__).resolve().parent
SYSTEMS = {
    # orbital files in ascending atomic number, as every LCAO model takes them
    "al": {"orbitals": ["Al_gga_8au_100Ry_4s4p1d.orb"], "spin": "none"},
    "gaas_soc": {"orbitals": ["Ga_gga_9au_60Ry_2s2p2d.orb", "As_gga_8au_60Ry_2s2p1d.orb"],
                 "spin": "spinorbit"},
}
FILES = {"abacus": "abacus/data-HR-sparse_SPIN0.csr", "nextham": "nextham/HR_atom_pairs.h5"}


def main(system, fmt):
    data = HERE / "data" / system
    config = SYSTEMS[system]
    kpoints = np.loadtxt(data / "abacus" / "kpoints.dat", ndmin=2).tolist()
    bands = np.loadtxt(data / "abacus" / "BANDS_1.dat", ndmin=2)[:, 2:]
    fermi = float((data / "abacus" / "E_Fermi_eV").read_text())

    model = LcaoModel(hamiltonian=HamiltonianFile(str(data / FILES[fmt]), format=fmt),
                      orbitals=OrbitalFiles(config["orbitals"], dir=str(data)), spin=config["spin"])
    structure = Structure.from_file(str(data / "conf.vasp"))
    calc = RSATB(structure, model, SetupOptions(kpoints=kpoints, compute_bounds=False,
                                                energy_bounds=[-200.0, 400.0]))
    def dense(which, k):
        # S has a nonzero diagonal, so the largest index fixes the dimension.
        rows, cols, vals = calc._cpp_obj.dump_operator_coo(which, 0, k)
        size = int(max(rows.max(), cols.max())) + 1
        return sp.coo_matrix((vals, (rows, cols)), shape=(size, size)).toarray()

    worst = worst_occupied = 0.0
    for k in range(len(kpoints)):
        e = la.eigh(dense("H", k), dense("S", k), eigvals_only=True)[:bands.shape[1]]
        err = np.abs(e - bands[k])
        worst = max(worst, err.max())
        worst_occupied = max(worst_occupied, err[bands[k] < fermi].max(initial=0.0))
    print("{} / {}: {} k-points x {} bands; max |e_rsatb - e_ABACUS| = {:.1e} eV "
          "(below E_F {:.1e} eV)".format(system, fmt, len(kpoints), bands.shape[1], worst,
                                         worst_occupied))


if __name__ == "__main__":
    main(*sys.argv[1:3])

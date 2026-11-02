"""Writes an ABACUS run's H(R) as a NextHam HR_atom_pairs.h5, with NextHam's own writer.

A NextHam prediction is post-processed into exactly this file; writing ABACUS's
own DFT H(R) through the same code gives a file of the same format whose values
are known, which is what the example uses when no trained model is at hand.

usage: python to_nextham.py --nextham /path/to/pre_post_process --abacus-out OUT.Al \
           --poscar conf.vasp --orbitals Al:Al_gga_8au_100Ry_4s4p1d.orb --nspin 1 --save-path DIR
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np


def spatial_orbitals(orb_file):
    """Orbital count of an ABACUS .orb file: sum over L of (2L + 1) x zeta count."""
    count = 0
    for line in pathlib.Path(orb_file).read_text().splitlines():
        words = line.split()  # "Number of Sorbital-->  2"
        if len(words) == 4 and words[:2] == ["Number", "of"] and words[2].endswith("orbital-->"):
            shell = "SPDFGH".index(words[2][0].upper())
            count += (2 * shell + 1) * int(words[3])
    if count == 0:
        raise ValueError("no 'Number of ...orbital' lines in " + str(orb_file))
    return count


def read_poscar(path):
    lines = pathlib.Path(path).read_text().splitlines()
    scale = float(lines[1])
    cell = np.array([[float(x) for x in lines[2 + i].split()[:3]] for i in range(3)]) * scale
    symbols, counts = lines[5].split(), [int(x) for x in lines[6].split()]
    species = [s for s, n in zip(symbols, counts) for _ in range(n)]
    coords = np.array([[float(x) for x in lines[8 + i].split()[:3]] for i in range(len(species))])
    positions = coords @ cell if lines[7].strip()[0] in "Dd" else coords * scale
    return species, positions, cell


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nextham", required=True, type=pathlib.Path, help="NextHam's pre_post_process directory")
    p.add_argument("--abacus-out", required=True, type=pathlib.Path)
    p.add_argument("--poscar", required=True, type=pathlib.Path, help="the structure, atoms in ABACUS order")
    p.add_argument("--orbitals", required=True, nargs="+", help="Symbol:file.orb per species")
    p.add_argument("--nspin", type=int, choices=(1, 4), required=True)
    p.add_argument("--save-path", required=True, type=pathlib.Path)
    args = p.parse_args()

    spec = importlib.util.spec_from_file_location("pair_hdf5", args.nextham / "pair_hdf5.py")
    pair_hdf5 = importlib.util.module_from_spec(spec)
    sys.modules["pair_hdf5"] = pair_hdf5
    spec.loader.exec_module(pair_hdf5)
    from ase.data import atomic_numbers

    norb = {}
    for item in args.orbitals:
        symbol, orb = item.split(":", 1)
        norb[symbol] = spatial_orbitals(orb)
    species, positions, cell = read_poscar(args.poscar)
    counts = np.array([norb[s] * (2 if args.nspin == 4 else 1) for s in species])
    layout = pair_hdf5.BasisLayout(counts, nspin=args.nspin, positions=positions, cell=cell,
                                   symbols=species,
                                   atomic_numbers=np.array([atomic_numbers[s] for s in species]))
    out = args.abacus_out
    hr = next(p for p in (out / "hrs1_nao.csr", out / "data-HR-sparse_SPIN0.csr") if p.exists())
    sr = next(p for p in (out / "srs1_nao.csr", out / "data-SR-sparse_SPIN0.csr") if p.exists())
    args.save_path.mkdir(parents=True, exist_ok=True)
    hr_h5, sr_h5 = pair_hdf5.import_abacus_pair_hdf5(
        hr, sr, args.save_path, nspin=args.nspin, hr_unit="Ry", layout=layout,
        database=args.save_path / "_pair_blocks.sqlite")
    (args.save_path / "_pair_blocks.sqlite").unlink(missing_ok=True)
    print("wrote", hr_h5, "and", sr_h5)


if __name__ == "__main__":
    main()

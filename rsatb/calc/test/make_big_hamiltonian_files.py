"""Writes a primitive-cell DeePTB H(R), folded onto an L x L x L cell, as ABACUS
text CSR, ABACUS binary CSR and NextHam HDF5.

The big-cell H(R) is exactly what tiling the primitive cell gives, so reading
these files untiled must reproduce the tiled primitive read block for block
(benchmark_hamiltonian_read.cpp). Blocks are written the way ABACUS and NextHam
store them: ABACUS orbital order (m = 0, +1, -1, ..., Condon-Shortley signs),
Rydberg in the CSR files, eV in the NextHam file. Values are printed with 17
significant digits, so the comparison is exact to rounding.

Atoms of the big cell are numbered as rsatb's Structure.BuildSupercell numbers
them: atom = ((ix * L + iy) * L + iz) * n_primitive + p.

usage: python make_big_hamiltonian_files.py --primitive hamiltonians.h5 \
           --shells 0,0,0,0,1,1,1,1,2 --supercell 12 --out DIR
"""
import argparse
import pathlib

import h5py
import numpy as np
import scipy.sparse as sp

EV_PER_RYDBERG = 13.605698  # ABACUS's Ry_to_eV, as the readers use


def abacus_order(shells):
    """rsatb index k -> (ABACUS index within the atom, sign), as in abacus_convention.hpp."""
    source, sign, offset = [], [], 0
    for l in shells:
        for m in range(-l, l + 1):
            source.append(offset + (0 if m == 0 else (2 * m - 1 if m > 0 else -2 * m)))
            sign.append(-1.0 if abs(m) % 2 else 1.0)
        offset += 2 * l + 1
    return np.array(source), np.array(sign)


def read_primitive(path, shells):
    """DeePTB blocks {(i, j, R): ABACUS-order block}, completed with (j, i, -R) partners."""
    source, sign = abacus_order(shells)
    blocks = {}
    with h5py.File(path, "r") as f:
        for name, data in f["0"].items():
            i, j, rx, ry, rz = map(int, name.split("_"))
            block = np.asarray(data, dtype=np.float64)
            if block.ndim == 3:
                block = block[..., 0]
            abacus = np.empty_like(block)
            # rsatb(a, b) = s_a s_b abacus(src_a, src_b)  =>  abacus = inverse.
            abacus[np.ix_(source, source)] = sign[:, None] * sign[None, :] * block
            blocks[(i, j, (rx, ry, rz))] = abacus
    for (i, j, r), block in list(blocks.items()):
        partner = (j, i, tuple(-x for x in r))
        if partner not in blocks:
            blocks[partner] = block.T.copy()
    return blocks


def fold(blocks, n_primitive, L, norb):
    """Big-cell entries: arrays I, J, Rbig (n, 3) and the primitive block index of each."""
    images = np.array(np.meshgrid(*[np.arange(L)] * 3, indexing="ij")).reshape(3, -1).T
    flat = lambda a: (a[:, 0] * L + a[:, 1]) * L + a[:, 2]
    keys = list(blocks)
    I, J, R, which = [], [], [], []
    for k, (p, q, r) in enumerate(keys):
        t = images + np.array(r)
        I.append(flat(images) * n_primitive + p)
        J.append(flat(t % L) * n_primitive + q)
        R.append(np.floor_divide(t, L))
        which.append(np.full(len(images), k))
    I, J, R, which = map(np.concatenate, (I, J, R, which))
    order = np.lexsort((R[:, 2], R[:, 1], R[:, 0], J, I))
    stack = np.stack([blocks[k] for k in keys])
    return I[order], J[order], R[order], which[order], stack


def write_csr(path_text, path_binary, I, J, R, which, stack, n, norb):
    shifts = sorted(set(map(tuple, R)))
    a = np.arange(norb)
    with open(path_text, "w") as text, open(path_binary, "wb") as binary:
        text.write("STEP: 0\nMatrix Dimension of H(R): {}\nMatrix number of H(R): {}\n".format(n, len(shifts)))
        np.array([0, n, len(shifts)], dtype=np.int32).tofile(binary)
        for r in shifts:
            sel = np.all(R == np.array(r), axis=1)
            rows = (I[sel][:, None, None] * norb + a[None, :, None]).repeat(norb, axis=2).ravel()
            cols = (J[sel][:, None, None] * norb + a[None, None, :]).repeat(norb, axis=1).ravel()
            vals = (stack[which[sel]] / EV_PER_RYDBERG).ravel()
            keep = vals != 0.0
            m = sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n))
            m.sort_indices()
            text.write("{} {} {} {}\n".format(*r, m.nnz))
            np.array([*r, m.nnz], dtype=np.int32).tofile(binary)
            if m.nnz == 0:
                continue
            for array, fmt in ((m.data, "%.17g"), (m.indices, "%d"), (m.indptr, "%d")):
                text.write(" ")
                array.tofile(text, sep=" ", format=fmt)
                text.write("\n")
            m.data.astype(np.float64).tofile(binary)
            m.indices.astype(np.int32).tofile(binary)
            m.indptr.astype(np.int32).tofile(binary)


def write_nextham(path, I, J, R, which, stack, n_atom, norb, z):
    pair_start = np.r_[True, (I[1:] != I[:-1]) | (J[1:] != J[:-1])]
    pair_of = np.cumsum(pair_start) - 1
    n_pair = pair_of[-1] + 1
    first = np.flatnonzero(pair_start)
    with h5py.File(path, "w") as f:
        f.attrs.update(format="abacus_atom_pair_operator_v1", operator="HR", HR_unit="eV",
                       unit="eV", nspin=np.int64(1), basis_num=np.int64(n_atom * norb),
                       atom_count=np.int64(n_atom))
        f["atom_basis_count"] = np.full(n_atom, norb, dtype=np.int64)
        f["atom_basis_start"] = np.arange(n_atom, dtype=np.int64) * norb
        f["atomic_numbers"] = np.full(n_atom, z, dtype=np.int64)
        f["pair_indices"] = np.stack([I[first], J[first]], axis=1).astype(np.int64)
        f["pair_offset"] = np.r_[first, len(I)].astype(np.int64)
        f["data_offset"] = (np.r_[first, len(I)] * norb * norb).astype(np.int64)
        f["atom_pair_offset"] = np.searchsorted(I[first], np.arange(n_atom + 1)).astype(np.int64)
        f["R"] = R.astype(np.int64)
        data = f.create_dataset("data", shape=(len(I) * norb * norb,), dtype=np.complex128)
        chunk = 1 << 16
        for s in range(0, len(I), chunk):
            e = min(len(I), s + chunk)
            data[s * norb * norb:e * norb * norb] = stack[which[s:e]].ravel().astype(np.complex128)
    return n_pair


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--primitive", required=True)
    p.add_argument("--shells", required=True, help="angular momenta of one atom's shells, e.g. 0,0,1,1,2")
    p.add_argument("--supercell", type=int, required=True)
    p.add_argument("--z", type=int, required=True, help="atomic number, for the NextHam file")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    shells = [int(x) for x in args.shells.split(",")]
    norb = sum(2 * l + 1 for l in shells)
    blocks = read_primitive(args.primitive, shells)
    n_primitive = 1 + max(max(i, j) for i, j, _ in blocks)
    L = args.supercell
    n_atom = n_primitive * L ** 3
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    I, J, R, which, stack = fold(blocks, n_primitive, L, norb)
    stem = "big_{}".format(L)
    write_csr(out / (stem + ".csr"), out / (stem + ".bin.csr"), I, J, R, which, stack, n_atom * norb, norb)
    n_pair = write_nextham(out / (stem + "_HR_atom_pairs.h5"), I, J, R, which, stack, n_atom, norb, args.z)
    print("{} atoms, {} blocks, {} pairs, {} R vectors -> {}".format(
        n_atom, len(I), n_pair, len(set(map(tuple, R))), out))


if __name__ == "__main__":
    main()

---
description: Read H(R) from DeePTB HDF5, ABACUS out_mat_hs2 CSR, or NextHam HR_atom_pairs.h5, and S(R) from DeePTB HDF5 — orbital files, atom order, supercells and spin-orbit coupling.
---

<p class="rs-kicker">Guide</p>

# Import DFT Hamiltonians

<div class="rs-lead" markdown>
An `lcao` model reads a real-space Hamiltonian $H(\mathbf R)$ computed by
another code, together with the numerical atomic orbitals it was written in.
The overlap is computed from the orbital files.
</div>

## The input

```json title="lcao.json"
{
  "structure": {"file": "STRU.vasp"},
  "model": {
    "type": "lcao",
    "hamiltonian": {"path": "hrs1_nao.csr", "format": "abacus"},
    "orbitals": {"names": ["Ga_gga_8au_100Ry_2s2p2d.orb", "As_gga_8au_100Ry_2s2p1d.orb"],
                 "dir": "orbitals"}
  },
  "run": {"kpoints": {"mesh": [4, 4, 4]}, "nsamples": 32},
  "tasks": [{"type": "dos", "dE": 0.1}]
}
```

Give one `.orb` file per species, **in order of increasing atomic number**.
The number of orbitals per atom and the cutoff radius are read from these
files.

## Formats

| `format` | File |
|---|---|
| `deeptb` (default) | DeePTB HDF5: `hamiltonians.h5`, or `predicted_hamiltonians.h5` for a model's prediction |
| `abacus` | ABACUS `out_mat_hs2` output: `hrs1_nao.csr` (older versions: `data-HR-sparse_SPIN0.csr`), text or binary |
| `nextham` | NextHam `HR_atom_pairs.h5` |

For large ABACUS systems, use binary output: it is read in parallel, while
text output is read by a single process.

## Atom order

ABACUS and NextHam files refer to atoms by their index in the ABACUS
calculation, so **the structure must list the atoms in the same order**.
RSATB does not reorder atoms.

!!! warning "Keep the atom order"
    When converting an ABACUS `STRU` to a POSCAR or CIF for
    `structure.file`, use a converter that preserves the atom order. Tools
    that group atoms by species change the order and the Hamiltonian no
    longer matches the structure.

## The overlap

By default the overlap is computed from the orbital files. A DeePTB file can
replace it, which is what a model trained on both matrices needs, since its
Hamiltonian belongs with its own overlap:

```json
"model": {
  "type": "lcao",
  "hamiltonian": {"path": "hamiltonians.h5", "format": "deeptb"},
  "overlap": {"path": "overlaps.h5", "format": "deeptb"},
  "orbitals": {"names": ["C_gga_7au_100Ry_2s2p1d.orb"], "dir": "orbitals"}
}
```

The orbital files are still required: they give the number of orbitals and the
cutoff radius of each species, and the position operator behind the velocity.
Blocks missing from the file are zero, except that every atom must have an
onsite block. Overlap files written by ABACUS or NextHam are not read.

## Models without orbital files { #no-orbitals }

A fitted tight-binding model with an overlap, such as DeePTB's `nnsk` or `mix`,
has $H(\mathbf R)$ and $S(\mathbf R)$ but no orbitals behind them. Read its
files with a `tb_file` model, which takes the basis from the input:

```json
"model": {
  "type": "tb_file",
  "hamiltonian": {"path": "predicted_hamiltonians.h5", "format": "deeptb"},
  "overlap": {"path": "predicted_overlaps.h5", "format": "deeptb"},
  "type_symbols": ["Mo", "S"],
  "type_norb": [9, 9],
  "cutoff_radius": [2.4, 2.4],
  "position": "mulliken"
}
```

Set `cutoff_radius` so that every pair in the file is within $r_i + r_j$: pairs
outside it are not read. Without orbitals the dipole integrals do not exist, so
without `position` the model has no velocity, and offers the DOS, PDOS,
occupation and Fermi level.

`"position": "mulliken"` takes $X_{ij}(\mathbf R) = \tfrac12(\mathbf r_i +
\mathbf R + \mathbf r_j)\, S_{ij}(\mathbf R)$, which adds the transport tasks.
Band velocities are then exact. The error of interband terms grows with the
transition energy, so `dc` and low-energy `msd` are its uses, and `ac` away
from low frequency is not reliable. Measured against true dipole integrals on
two DFT bases, the interband $|v|^2$ was within 12% at 1–3 eV for GaAs and a factor
of 2 too large for Al, and far too large above 10 eV in both.

## Supercells

`structure.supercell` repeats the primitive cell, and the primitive-cell
Hamiltonian is mapped onto the supercell when it is read. Only the
primitive-cell Hamiltonian file is needed:

```json
"structure": {"file": "conf.vasp", "supercell": [8, 8, 8]}
```

## Spin-orbit coupling { #spin-orbit }

Hamiltonians from spin-orbit calculations (`nspin = 4`) are read with
`"spin": "spinorbit"`, in all three formats:

```json
"model": {
  "type": "lcao",
  "hamiltonian": {"path": "hrs1_nao.csr", "format": "abacus"},
  "orbitals": {"names": ["Ga.orb", "As.orb"], "dir": "orbitals"},
  "spin": "spinorbit"
}
```

The Hamiltonian is then complex and the basis includes spin. All spectral
and transport tasks are available; the real-space tasks `ldos` and `density`
are not. A `deeptb` file holds blocks of twice the orbitals, with complex
values, and its overlap file, if there is one, holds the real blocks of the
spatial orbitals. An orthogonal DeePTB Hamiltonian with spin-orbit coupling is
read the same way by a `tb_file` model, whose `type_norb` then counts both spins.

The `sktb` and `dftb` models have no spin-orbit parameters, so they cannot use a
spinor basis.

## Checking an import

Before a production run, compare a small cell with the code that produced the
Hamiltonian:

1. Run `energy_bounds` and `dos` for the primitive cell on a k-mesh. Band
   edges and gaps should match the DFT band structure.
2. If they do not, the atom order or the order of the orbital files is the
   most common cause.

[Tutorial 7](../tutorials/abacus-nextham.md) does this comparison for fcc
Al and spin-orbit GaAs with `examples/abacus_nextham/check_bands.py`, which
compares the bands of the imported $H$ and $S$ with ABACUS's own
`BANDS_1.dat`.

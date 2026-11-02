---
description: Read H(R) from DeePTB HDF5, ABACUS out_mat_hs2 CSR, or NextHam HR_atom_pairs.h5 — orbital files, atom order, supercells and spin-orbit coupling.
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
| `deeptb` (default) | DeePTB HDF5 |
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

The overlap matrix is always computed from the orbital files; overlap files
written by ABACUS or NextHam are not used.

## Supercells

`structure.supercell` repeats the primitive cell, and the primitive-cell
Hamiltonian is mapped onto the supercell when it is read. Only the
primitive-cell Hamiltonian file is needed:

```json
"structure": {"file": "conf.vasp", "supercell": [8, 8, 8]}
```

## Spin-orbit coupling { #spin-orbit }

Hamiltonians from spin-orbit calculations (ABACUS or NextHam with
`nspin = 4`) are read with `"spin": "spinorbit"`:

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
are not.

## Checking an import

Before a production run, compare a small cell with the code that produced the
Hamiltonian:

1. Run `energy_bounds` and `dos` for the primitive cell on a k-mesh. Band
   edges and gaps should match the DFT band structure.
2. If they do not, the atom order or the order of the orbital files is the
   most common cause.

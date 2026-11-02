---
description: Read H(R) from DeePTB HDF5, ABACUS out_mat_hs2 CSR, or NextHam HR_atom_pairs.h5 — orbital order, atom order, supercells and spin-orbit.
---

<p class="rs-kicker">How-to guide</p>

# Import DFT Hamiltonians

<div class="rs-lead" markdown>
An `lcao` model reads a precomputed real-space Hamiltonian $H(\mathbf R)$ and
computes the overlap from the numerical atomic orbitals it was written in.
Three file layouts are supported, chosen by `hamiltonian.format`.
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

One `.orb` file per species, **in ascending atomic-number order**. The orbital
count per atom and the graph cutoff are read from them; nothing about the
basis is declared.

## Formats

| `format` | File | Notes |
|---|---|---|
| `deeptb` (default) | DeePTB HDF5: one group `"0"` holding datasets `i_j_Rx_Ry_Rz` | read in parallel |
| `abacus` | ABACUS `out_mat_hs2` sparse H(R): `hrs1_nao.csr` (older versions: `data-HR-sparse_SPIN0.csr`), text or binary, in Rydberg | binary is read in parallel; text is parsed by rank 0 in one streaming pass |
| `nextham` | NextHam post-processed `HR_atom_pairs.h5` | read in parallel |

Every format is read *distributed*: each MPI rank requests exactly the blocks
of the atoms it owns and holds nothing else. Measured on a 1,728-atom
aluminium file with 4.3 × 10⁷ non-zeros:

| Format | 1 rank | 16 ranks |
|---|---:|---:|
| NextHam HDF5 | 2.5 s | 0.23 s |
| ABACUS binary CSR | 3.0 s | 0.20 s |
| ABACUS text CSR | ~20 s | ~20 s |

Prefer binary output from ABACUS for large systems.

## Orbital and atom order

ABACUS and NextHam files use ABACUS's orbital order and sign convention, which
RSATB converts to its own using the shells listed in the `.orb` files — so
those two formats need an `lcao` model. The **structure must list atoms in the
same order as the ABACUS calculation**: atom indices in the file refer to that
order, and RSATB never reorders coordinates.

!!! warning "Keep the atom order"
    When converting an ABACUS `STRU` to a POSCAR or CIF for
    `structure.file`, use a converter that preserves atom order. Tools that
    group atoms by species will silently scramble the Hamiltonian's indexing.

The overlap is always recomputed from the orbitals; ABACUS's `srs1_nao.csr`
and NextHam's `SR_atom_pairs.h5` are not read.

## Supercells from a primitive Hamiltonian

`structure.supercell` tiles the primitive cell during setup, and the primitive
$(i, j, \mathbf R)$ blocks are mapped onto the expanded cell while the file is
read. No tiled file is ever written, so a 10⁶-atom supercell needs only the
primitive-cell Hamiltonian on disk:

```json
"structure": {"file": "conf.vasp", "supercell": [8, 8, 8]}
```

## Spin-orbit { #spin-orbit }

Read an nspin = 4 (spin-orbit) ABACUS or NextHam file with
`"spin": "spinorbit"` on the model:

```json
"model": {
  "type": "lcao",
  "hamiltonian": {"path": "hrs1_nao.csr", "format": "abacus"},
  "orbitals": {"names": ["Ga.orb", "As.orb"], "dir": "orbitals"},
  "spin": "spinorbit"
}
```

$H(\mathbf R)$ is then complex and the basis is spinor, with spin interleaved
inside each orbital (index $2\times\text{orbital} + s$). The overlap and
position operators become $S\otimes I_2$ and $r\otimes I_2$, and transport
prefactors drop the spin-degeneracy factor of 2 that a spin-degenerate basis
uses. The real-space `ldos` and `density` tasks are spin-degenerate only.

## Checking an import

Before any production run, compare a small cell against the code that wrote
the Hamiltonian:

1. Run `energy_bounds` and `dos` on the primitive cell with a k-mesh; the
   band edges and gap should match the DFT band structure.
2. If the gap is wrong or the window is absurd, the atom order or orbital file
   order is the usual cause.

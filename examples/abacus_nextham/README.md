# ABACUS and NextHam Hamiltonians

rsatb reads two more H(R) sources besides DeePTB: ABACUS's own sparse H(R)
(`format: "abacus"`) and NextHam's post-processed prediction
(`format: "nextham"`). Both are LCAO Hamiltonians in ABACUS's orbital order, so
both go into an `lcao` model together with the `.orb` files the ABACUS data was
computed with; rsatb converts the orbital order, the units and the spin layout
on read and builds S from the orbital files.

```bash
python check_bands.py al abacus                 # did rsatb read the file right?
mpirun -n 8 rsatb run al_abacus.json            # fcc Al: DOS, Fermi level, dc conductivity
mpirun -n 8 rsatb run al_nextham.json           # the same, from a NextHam file
mpirun -n 8 rsatb run gaas_soc_nextham.json     # GaAs with spin-orbit coupling
```

| file | what it shows |
|---|---|
| `al_abacus.json` | ABACUS's H(R) (`data-HR-sparse_SPIN0.csr`) for fcc Al, spin-degenerate |
| `al_nextham.json` | the same H(R) as a NextHam `HR_atom_pairs.h5` |
| `gaas_soc_nextham.json` | an nspin=4 (spin-orbit) H(R): `spin: "spinorbit"`, complex H, spinor basis |
| `check_bands.py` | rsatb's bands against ABACUS's `BANDS_1.dat`, at ABACUS's own k-points |
| `make_data.sh` | regenerates `data/`: runs ABACUS (`abacus/*/`), converts to NextHam |
| `to_nextham.py` | writes any ABACUS H(R) as a NextHam file, with NextHam's own writer |

## Getting the Hamiltonians

**From ABACUS.** Any LCAO run with `out_mat_hs2 1` (see `abacus/al/INPUT`)
writes the sparse H(R) of the final step into `OUT.<suffix>/` --
`data-HR-sparse_SPIN0.csr` in ABACUS 3.10, `hrs1_nao.csr` in newer versions;
point `path` at it. Text and the older binary layout are both read. For a
spin-orbit run (`nspin 4`, `lspinorb 1`) set `spin: "spinorbit"`; the reader
expects the complex `(re,im)` values such a run writes, and refuses a file
whose spin does not match.

**From NextHam.** NextHam predicts H(R) from a cheap ABACUS input Hamiltonian
H0, in three steps of its `pre_post_process` scripts:

1. `pre_process.py --read-path <ABACUS run> --save-path <dir> --dataset-path <dir> --nspin 1|4`
   reads the structure and the ABACUS H0 matrices (`hrs1_nao.csr`, the
   run's `OUT.ABACUS/`) into `input_inference.pth`. Which ABACUS settings
   produce H0 is NextHam's choice; see its documentation.
2. NextHam's `infer.py`, with a trained model, writes the prediction PTH.
3. `post_process.py --prediction-path <pred.pth> --stru-file <STRU> --data-dir <ABACUS OUT dir> --save-path <dir> --nspin 1|4 --convert-only`
   writes `HR_atom_pairs.h5` (and `SR_atom_pairs.h5`). Point `path` at the
   first; rsatb does not read the second -- S comes from the orbital files.

Without a trained model, `to_nextham.py` writes ABACUS's own DFT H(R) through
the same NextHam writer, which gives a file of exactly the NextHam format with
known values -- that is what `data/*/nextham/` holds, so the two formats can
be checked against each other and against ABACUS.

## Rules a file has to meet

- **Atom order.** The structure must list the atoms in ABACUS's order (the
  STRU order); the file has no way to say otherwise. `data/*/conf.vasp` are
  written that way -- note As before Ga in GaAs.
- **Orbital files.** The `.orb` files must be the ones ABACUS used, given in
  ascending atomic number (Ga before As), as every `lcao` model takes them.
  NextHam files record each atom's orbital count and Z, and a mismatch is an
  error; ABACUS files record only the total, which is checked.
- **Spin.** nspin=1 is `spin: "none"`, nspin=4 is `spin: "spinorbit"`. The
  spinor basis interleaves spin inside each orbital, as ABACUS and NextHam do.
- **Units.** ABACUS's Rydberg and NextHam's declared unit are converted to eV.

## What the data reproduces

`check_bands.py` diagonalizes rsatb's H(k), S(k) at ABACUS's own SCF k-points:

| system / file | k-points x bands | max error, bands below E_F | max error, all bands |
|---|---|---|---|
| Al / abacus  | 868 x 16 | 1.0e-4 eV | 0.10 eV |
| Al / nextham | 868 x 16 | 1.0e-4 eV | 0.10 eV |
| GaAs SOC / nextham | 216 x 38 | 6.1e-5 eV | 5.4e-3 eV |
| GaAs SOC / abacus  | 216 x 38 | 6.1e-5 eV | 5.4e-3 eV |

The error grows with band energy -- 1e-7 eV for Al's core-like bands, 0.1 eV
at +72 eV -- and is the same with ABACUS's own S(R) in place of rsatb's
(they agree to 8e-7), so it lies between the H(R) ABACUS writes and the bands
it prints, not in the reading. GaAs's spin-orbit split-off gap at Gamma comes
out 0.349 eV (experiment: 0.34 eV).

The calculations themselves, on 8 ranks with the default preconditioner
(`operator_form: "solve"`, `maxiter: 0` -- the seed-only route that scales to
large systems), timed on a shared, loaded node:

| input | time | result |
|---|---|---|
| `al_abacus.json` | 9.5 min | mu = 11.64 eV (exact for this 6x6x6 sampling: 11.547 eV; ABACUS's own 12x12x12: 11.021 eV); sigma_dc = 1.84 / 1.94 / 2.02 MS/m (xx / yy / zz) at 300 K |
| `al_nextham.json` | 7.7 min | the same: DOS equal to 3.5e-9, same mu and sigma_dc |
| `gaas_soc_nextham.json` | 3.0 min | 144.4 electrons at mid-gap (exact: 144 -- a spinor basis counts both spins); Re sigma_xx(omega) 0 below the gap, 0.18 MS/m at 0.93 eV, 0.77 MS/m at 1.86 eV |

The inputs are sized to run in minutes: Al's 8 au orbitals give each atom
~200 neighbours and a 415 eV spectral window, so cell size and resolution are
what cost. Raise `supercell`, the k-mesh, `nsamples` and the resolutions
(`dE`, `n_moments`, `nt`) for production numbers; mu from a coarse sampling of
a metal moves with the sampling, which is why `dc` uses the Fermi level of its
own sampling, not ABACUS's.

## Scaling to large systems

Each MPI rank reads only the blocks of its own atoms. NextHam files and
ABACUS's binary CSR are read in parallel, each rank reading only its rows:
memory per rank is the rank's share, and read time drops with the rank count
(a 1728-atom Al file: 2.5 s on 1 rank, 0.23 s on 16). ABACUS's text CSR -- the
only form ABACUS 3.10 writes -- cannot be read out of order: rank 0 parses it
in one pass and holds one R record at a time, which for a single large cell is
most of the matrix. For systems beyond ~10^4 atoms, read a NextHam file, or
convert the ABACUS output once with `to_nextham.py`.

## Not yet supported

Spinor (spin-orbit) runs do the spectral and transport tasks; the real-space
`ldos` and `density` maps are spin-degenerate only. DeePTB spinor files are
not read.

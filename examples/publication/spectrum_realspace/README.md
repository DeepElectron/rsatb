# Spectrum and real-space maps (the gallery figure)

The nine runs, the diagonalization references and the figure script behind
the spectrum / density / LDOS gallery of the paper: rows are diamond and
silicon (12×12×12 of the 2-atom cell, 3,456 atoms, 44,928 basis functions)
and aluminium (12×12×12 of the 1-atom cell, 1,728 atoms, 36,288 basis
functions); columns are the DOS and PDOS, the charge density, two LDOS maps
and a line cut, each against the exact result on the same cell.

```bash
pip install .                              # also builds the diag_kpoint reference tool
OMP_NUM_THREADS=48 ./run_production.sh     # the nine runs, hours each, ~100 GB peak
./make_references.sh                       # k-folded diagonalization, minutes
python plot_ldos_gallery.py                # fig_ldos_gallery.png / .pdf
python compare_spectrum.py                 # DOS/PDOS deviations, both error sources split
python compare_ldos.py                     # LDOS deviation per energy, with the realised capture
```

| panel | input | task | reference |
|---|---|---|---|
| DOS, PDOS | `{dia,si,al}_pdos.json` | `pdos`, dE 0.3 eV, 128 samples | `results/*_rs_k12` (same cell), `results/*_k24` (converged) |
| density | `{dia,si,al}_density.json` | `density`, fragment surrogate, 64 samples | `results/*_rs_k12` |
| LDOS | `{dia,si,al}_ldos.json` | `ldos`, survey estimator, 64 probes × 12 families, η 0.25 eV | `results/*_rs_k12` |

Every input reads the primitive cells in `test/data/` and tiles them at load
(`structure.supercell`); the reference diagonalizes the same primitive cell on
the same mesh and folds it, so a 12³ k-mesh is the 12³ supercell at Γ by band
folding and the quoted deviation is the estimator's alone. The Fermi levels in
the inputs come from the occupation search (diamond, silicon) and from the
k-point fold with 11 electrons per atom (aluminium, μ = 11.112081 eV); the
LDOS energies of aluminium are E_F − 4, E_F, E_F + 4 eV, those of the gapped
systems sit in the valence band, at a band edge and in the conduction band.

`kpm_ref.py` rebuilds the reference DOS through the run's own Jackson kernel
(moment count, edge inset, amplitude convention), which is what makes the
comparison fair at van Hove singularities. `diag_kpoint.cpp` is the reference
tool itself (CMake target `diag_kpoint`); its real-space fold is self-checked
against the supercell overlap and stops rather than write a plausible wrong map.

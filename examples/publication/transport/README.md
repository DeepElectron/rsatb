# Linear response and transport (the transport figure)

The three runs, the two references and the figure script behind the transport
figure of the paper: rows are diamond and silicon (8×8×8 of the 2-atom cell,
1,024 atoms) and aluminium (9×9×9 of the 1-atom cell, 729 atoms); columns are
the Kubo-Greenwood conductivity σ(E) with the finite-temperature dc point at
E_F, the optical conductivity σ(ω) (real and imaginary parts) and the
ballistic diffusion D(E, t) at three energies. Every curve is the 64-sample
stochastic estimator with the commutator velocity at Γ; the quantum metric is
computed in the same run.

```bash
OMP_NUM_THREADS=48 ./run_production.sh     # diamond, silicon: about a day each; aluminium: est. a week
./make_references.sh                       # same-cell (minutes) and bulk (hours) references
python plot_transport.py                   # fig_transport.png / .pdf
```

One input per system, five tasks from one setup: `dc_spectrum`, `dc`, `ac`
(600 time steps, concurrent evolution), `msd` (80 × 0.5 fs, spectral route,
no moment prediction) and `quantum_metric`. Aluminium's 4s4p1d basis spans
~640 eV, four times the silicon and diamond cells, so its moment counts are
four times theirs for the same energy resolution, and its `dc` tasks carry
`max_workspace_mb` to partition the FFT-Chebyshev transform. For a metal the
`ac` Fermi operator must be sized to kT (16,000 moments here, ~0.06 eV against
26 meV); the default is far too coarse.

Two references sit under every curve. The pale grey band is the trace limit
from dense diagonalization of the same supercell with the production
operators (`gen_reference_ops.py` → `make_ref_curves.py`): same window, same
Jackson kernel, same truncated Fermi operator, so it measures sampling error
alone. The pale warm band is the bulk limit (`make_bulk_ref.py`): the
primitive cell on a 24³ k-mesh, the textbook Kubo sums with a Gaussian of the
kernel's measured width, the Drude weight reported separately because the
code's optical conductivity is the regular part. Every quantity is the
spin-degenerate SI value.

Measured on the diamond and silicon runs: the sampling error is 1–3 %
(median over significant points); the distance to the bulk is the finite
cell, up to 30–45 % pointwise for a 1,024-atom cell at 0.15 eV resolution
while the integrated spectral weights agree to 3–10 %. The aluminium input
is the 9×9×9 run of the 1-atom cell; at the time of writing it had not been
run to completion here (an earlier 3×3×3 run of a superseded 27-atom model
stood in for the metal row), so its cost above is an estimate and its
deviations are not quoted.

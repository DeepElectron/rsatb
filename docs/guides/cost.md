---
description: Estimate the number of matrix-vector products an RSATB input requires before submitting it, with examples/estimate_matvecs.py.
---

<p class="rs-kicker">Guide</p>

# Estimate cost first

<div class="rs-lead" markdown>
Most of the run time is spent in sparse matrix–vector products. Their number
follows from the Chebyshev expansions of the requested tasks and can be
computed from the input before the run.
</div>

```console
$ python examples/estimate_matvecs.py examples/06_matblg_transport.json
  dos                      19,456 matvecs   1024 time steps x 19 Chebyshev terms
  msd                     839,398 matvecs   KPM density 706798 + affine time 132600
  quantum_metric           94,886 matvecs   H0 91808 + velocity 3078
  total                   953,740
```

The estimate needs the spectral window:

- from the input, if `run.bounds.mode` is `"fixed"`;
- otherwise from `--bounds EMIN EMAX`,
- or from `--summary path/to/summary.json` of an earlier run of the same system.

The script requires only Python.

## From matrix-vector products to run time

Measure the time per product in a short run of the same system (one task,
few moments), then:

$$
t_{\text{run}} \approx n_{\text{matvec}} \times n_{\text{samples}} \times n_k
\times t_{\text{matvec}} / n_{\text{groups}}.
$$

For non-orthogonal models one product of the expansion operator contains
several sparse products, so measure $t_{\text{matvec}}$ with the same
`operator_form` as the production run.

## Reducing the cost

1. **Coarser resolution,** where the analysis allows: the cost is linear in
   the number of moments.
2. **A fixed spectral window** (`bounds.mode = "fixed"`) skips the
   eigenvalue estimate; the window must still contain the whole spectrum.
3. **Fewer random vectors for larger cells:** the relative statistical error
   decreases with system size.
4. **Several tasks in one input:** they share one setup.

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
$ python examples/estimate_matvecs.py examples/tutorials/diamond_optics.json --bounds -8.6 264
  dos                       3,601 matvecs
  ac                        6,242 matvecs
  total                     9,843
```

The counts are per random vector and per k-point (output abridged).

The estimate needs the spectral window:

- from the input, if `run.bounds.mode` is `"fixed"`;
- otherwise from `--bounds EMIN EMAX`,
- or from `--summary path/to/summary.json` of an earlier run of the same system.

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

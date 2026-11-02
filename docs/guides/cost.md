---
description: Estimate the matrix-vector product count of an RSATB input before queueing it, with examples/estimate_matvecs.py.
---

<p class="rs-kicker">How-to guide</p>

# Estimate cost first

<div class="rs-lead" markdown>
A run spends its time on sparse matrix–vector products, and their number
follows from the Chebyshev expansions its tasks use. It can be worked out from
the input alone, before anything is queued.
</div>

```console
$ python examples/estimate_matvecs.py examples/06_matblg_transport.json
  dos                      19,456 matvecs   1024 time steps x 19 Chebyshev terms
  msd                     839,398 matvecs   KPM density 706798 + affine time 132600
  quantum_metric           94,886 matvecs   H0 91808 + velocity 3078
  total                   953,740
```

The estimator needs the spectral window, because several expansions are
truncated against it:

- an input with `run.bounds.mode = "fixed"` states it;
- otherwise pass `--bounds EMIN EMAX`,
- or `--summary path/to/summary.json` from an earlier run of the same system.

Like `rsatb check`, it imports nothing compiled, so it runs on a login node.

## From matvecs to wall time

Time a short run of the same system — one task, a few moments — and read the
profile table. The time per matvec is then roughly constant:

$$
t_{\text{run}} \approx n_{\text{matvec}} \times n_{\text{samples}} \times n_k
\times t_{\text{matvec}} / n_{\text{groups}}.
$$

For a non-orthogonal model, one "matvec" of the expansion operator is several
sparse products — two $Z$ products and one $H$ product in the sandwich form,
or an $H$ product plus a conjugate-gradient solve in the solve form — so
$t_{\text{matvec}}$ must be measured on the same `operator_form`.

## Levers, cheapest first

1. **Coarser resolution** where the physics allows: cost is linear in moments.
2. **A narrower window**: with `bounds.mode = "fixed"` and a known spectrum
   you skip the PRIMME estimate, though the window must still contain every
   eigenvalue.
3. **Fewer samples on bigger cells**: self-averaging pays for size.
4. **One setup, many tasks**: tasks in the same input share the setup.

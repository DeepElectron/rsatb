---
description: The numerical methods behind RSATB — stochastic traces, Chebyshev expansions, non-orthogonal bases and linear response.
---

<p class="rs-kicker">Theory</p>

# How RSATB works

<div class="rs-lead" markdown>
Background on the numerical methods. The tutorials and guides do not
depend on these pages; they explain what the main settings control.
</div>

!!! tip "Summary"
    RSATB does not diagonalize the Hamiltonian. It applies it repeatedly to a
    small number of random vectors through Chebyshev recurrences and averages
    the results. For a sparse Hamiltonian the cost is linear in the number of
    atoms.

The observables RSATB computes are traces, or diagonals, of matrix functions
of the Hamiltonian. The pages below cover how these are estimated, and how
the same approach extends to response functions.

$$
\rho(E) = \operatorname{Tr}\,\delta(E-\hat H), \qquad
n(\mathbf r) = \langle \mathbf r|\,f_F(\hat H)\,|\mathbf r\rangle, \qquad
\sigma_{\alpha\beta} \propto \operatorname{Tr}\big[\hat v_\alpha\,\delta(E-\hat H)\,\hat v_\beta\,\delta(E-\hat H)\big].
$$

<div class="grid cards" markdown>

-   **[Random-state traces](stochastic-trace.md)**

    ---

    A trace is estimated as the average of $\langle\xi|A|\xi\rangle$ over
    random vectors; the relative error decreases with system size.

-   **[Chebyshev expansions](chebyshev.md)**

    ---

    Matrix functions are expanded in polynomials and applied by a three-term
    recurrence of sparse matrix–vector products.

-   **[Non-orthogonal bases](overlap.md)**

    ---

    LCAO and DFTB overlaps enter through $S^{-1}H$ or $S^{-1/2}HS^{-1/2}$,
    with a sparse, graph-local inverse square root.

-   **[Linear response](linear-response.md)**

    ---

    Kubo formulas for DC and optical conductivity, and the wave-packet
    picture of diffusion, built from the commutator velocity.

</div>

!!! note "Design notes"
    The derivations, measurements and rejected alternatives behind each
    algorithm live in the repository's
    [`doc/`](https://github.com/DeepElectron/rsatb/tree/main/doc) directory
    — for example the survey-sampling LDOS estimator, the spectral MSD route
    and the inverse-square-root memory model. These pages are the overview.

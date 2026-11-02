---
description: The numerical methods behind RSATB — stochastic traces, Chebyshev expansions, non-orthogonal bases and linear response.
---

<p class="rs-kicker">Theory</p>

# How RSATB works

<div class="rs-lead" markdown>
Every observable RSATB computes is a trace of a matrix function of the
Hamiltonian, or a diagonal of one. Three ideas make those traces affordable at
millions of orbitals; a fourth carries them to response functions.
</div>

$$
\rho(E) = \operatorname{Tr}\,\delta(E-\hat H), \qquad
n(\mathbf r) = \langle \mathbf r|\,f_F(\hat H)\,|\mathbf r\rangle, \qquad
\sigma_{\alpha\beta} \propto \operatorname{Tr}\big[\hat v_\alpha\,\delta(E-\hat H)\,\hat v_\beta\,\delta(E-\hat H)\big].
$$

<div class="grid cards" markdown>

-   **[Random-state traces](stochastic-trace.md)**

    ---

    A trace is the average of $\langle\xi|A|\xi\rangle$ over random vectors.
    Its relative error falls with system size, so large systems need *fewer*
    samples.

-   **[Chebyshev expansions](chebyshev.md)**

    ---

    Matrix functions become polynomials applied by a three-term recurrence:
    only sparse matrix–vector products, never a factorization.

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

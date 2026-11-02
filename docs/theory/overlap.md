---
description: How RSATB handles a non-orthogonal overlap matrix — the S^-1 H and Z H Z operator forms, the graph-local S^-1/2 factor, and its cost.
---

<p class="rs-kicker">Theory</p>

# Non-orthogonal bases

<div class="rs-lead" markdown>
Atomic-orbital bases — LCAO from DFT, DFTB — overlap: $\langle i|j\rangle =
S_{ij} \neq \delta_{ij}$. The eigenproblem becomes generalized, and every
expansion has to act on the right operator.
</div>

## Two equivalent operators

The generalized problem $HC = ESC$ has the same spectrum as two ordinary ones:

$$
\big(S^{-1}H\big)\,C = E\,C
\qquad\text{and}\qquad
\big(S^{-1/2} H S^{-1/2}\big)\,S^{1/2}C = E\,S^{1/2}C .
$$

Traces are invariant under the similarity transformation, so either operator
gives the same DOS, occupation and response:
$\operatorname{Tr} f(\hat H) = \operatorname{Tr} f(S^{-1}H) =
\operatorname{Tr} f(S^{-1/2}HS^{-1/2})$. Writing $Z = S^{-1/2}$, RSATB offers
both forms through `run.preconditioner.operator_form`:

| | `"solve"` (default) | `"sandwich"` |
|---|---|---|
| Operator applied | $S^{-1}H$ | $ZHZ$ |
| One application | $H$ product + a preconditioned CG solve with $S$ | two $Z$ products and one $H$ product |
| Hermitian | no — moments cost one product each | yes — two moments per product |
| Memory | keeps $S$ | releases $S$ after setup, keeps $Z$ |
| PDOS convention | Mulliken | Löwdin |
| Use when | $Z$ would be too dense to store | $Z$ fits in memory — usually the faster route |

## The graph-local $S^{-1/2}$

$S$ is sparse because orbitals have finite range, but its exact inverse
square root is not. RSATB builds $Z$ in two stages:

1. **A graph-local seed.** For each atom, the dense inverse square root of the
   overlap restricted to its neighbourhood in the atomic graph gives that
   atom's rows of $Z$. The seed is sparse by construction and accurate on the
   subgraph.
2. **Newton–Schulz refinement.** Coupled iterations
   $Z \leftarrow \tfrac12 Z(3I - Z S Z)$ drive $ZSZ \to I$ to the tolerance
   `preconditioner.tol`, dropping entries below `sptol`, for at most
   `maxiter` steps.

`maxiter = 0` stops at the seed. Under the `solve` form that is the
*seed-only* route: the sparse seed serves as a CG preconditioner for $S^{-1}$
at every product, cheap to store and scalable.

!!! warning "Cost and memory"
    The seed's dense neighbourhood blocks are the setup's memory peak, and they
    grow as (neighbours × orbitals per atom)² per thread. Rich bases — triple-ζ
    with polarization — on many threads can need tens of GB in this phase alone.
    Check the `setup/isqrt_*` rows of the profile table, and reduce
    `OMP_NUM_THREADS` before reducing accuracy.

## Conditioning

LCAO bases with diffuse or polarization orbitals are ill-conditioned:
$\kappa(S)$ of $10^4$ is common. The small eigenvalues of $S$ push the
largest eigenvalues of $S^{-1}H$ upward, which is why an LCAO spectral window
often extends hundreds of eV above the Fermi level, and why
[resolution is expensive](chebyshev.md#kernels-and-resolution) in such bases.

## Velocity in a non-orthogonal basis

The velocity operator is the commutator $\hat v_\alpha = i[\hat H, \hat
r_\alpha]/\hbar$. In a non-orthogonal basis it needs the position matrix
$\langle i|\hat r_\alpha|j\rangle$ — the dipole integrals — as well as $H$ and
$S$:

$$
v_\alpha = \frac{i}{\hbar}\,S^{-1}\big(H S^{-1} r_\alpha - r_\alpha S^{-1} H\big).
$$

An `lcao` model computes the dipole integrals from its numerical orbitals, so
it supplies velocity unconditionally. A non-orthogonal `sktb` or `dftb` model
has no orbitals to integrate, so it has neither position nor velocity, and the
transport tasks are unavailable for it.

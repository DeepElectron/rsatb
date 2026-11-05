---
description: How RSATB handles a non-orthogonal overlap matrix — the S^-1 H and Z H Z operator forms, the graph-local S^-1/2 factor, and its cost.
---

<p class="rs-kicker">Theory</p>

# Non-orthogonal bases

<div class="rs-lead" markdown>
In atomic-orbital bases, as in LCAO from DFT and in DFTB, orbitals on
neighbouring atoms overlap: $\langle i|j\rangle = S_{ij} \neq \delta_{ij}$.
The eigenproblem is then generalized, and the expansions must use an
operator that includes $S$.
</div>

## Two equivalent operators

The generalized problem $HC = ESC$ has the same spectrum as two ordinary ones:

$$
\big(S^{-1}H\big)\,C = E\,C
\qquad\text{and}\qquad
\big(S^{-1/2} H S^{-1/2}\big)\,S^{1/2}C = E\,S^{1/2}C .
$$

Traces do not change under a similarity transformation, so both operators
give the same DOS, occupation and response:
$\operatorname{Tr} f(\hat H) = \operatorname{Tr} f(S^{-1}H) =
\operatorname{Tr} f(S^{-1/2}HS^{-1/2})$. With $Z = S^{-1/2}$, RSATB offers
both forms through `run.preconditioner.operator_form`:

| | `"solve"` (default) | `"sandwich"` |
|---|---|---|
| Operator applied | $S^{-1}H$ | $ZHZ$ |
| One application | one $H$ product and a preconditioned CG solve with $S$ | two $Z$ products and one $H$ product |
| Hermitian | no; one moment per product | yes; two moments per product |
| Memory | keeps $S$ | keeps $Z$; $S$ is released after setup |
| PDOS convention | Mulliken | Löwdin |
| Use when | $Z$ is too dense to store | $Z$ fits in memory |

## The graph-local $S^{-1/2}$

$S$ is sparse because orbitals have a finite range, but its exact inverse
square root is not. RSATB builds $Z$ in two stages:

1. **A graph-local approximation.** For each atom, the inverse square root of
   the overlap restricted to the atom's neighbourhood gives that atom's rows
   of $Z$. This approximation is sparse.
2. **Newton–Schulz refinement.** The iteration
   $Z \leftarrow \tfrac12 Z(3I - Z S Z)$ brings $ZSZ$ towards $I$, to the
   tolerance `preconditioner.tol`, drops entries smaller than `sptol`, and
   stops after at most `maxiter` steps.

With `maxiter = 0` the refinement is skipped. In the `solve` form the
approximation is then used as the preconditioner of a conjugate-gradient
solve with $S$ at every product.

This combination, `solve` with `maxiter = 0`, is the default. The
approximation stays sparse, so its cost and memory grow linearly with the
number of atoms. Refinement makes $Z$ denser; for diamond with a few thousand
atoms the refined $Z$ is dense. Where the refined $Z$ fits in memory,
`"sandwich"` with `maxiter = 20` is faster per product (5.9 times for
4096-atom diamond, `examples/08_lcao_diamond888_dos_ac.json`). The sandwich
form needs the refinement, because it applies $Z$ itself and the error of an
unrefined $Z$ would enter the results; `rsatb check` warns about this
combination.

!!! warning "Memory"
    The memory peak of the setup is set by the dense neighbourhood blocks of
    the approximation. It grows with the number of orbitals per atom and with
    the number of threads per process. For large basis sets, use fewer
    threads per process.

## Conditioning

LCAO bases with diffuse or polarization orbitals are ill-conditioned; a
condition number $\kappa(S)$ of $10^4$ is common. The small eigenvalues of
$S$ raise the largest eigenvalues of $S^{-1}H$. This is why an LCAO spectral
window often extends hundreds of eV above the Fermi level, and why
[resolution is expensive](chebyshev.md#kernels-and-resolution) in such bases.

## Velocity in a non-orthogonal basis

The velocity operator is the commutator $\hat v_\alpha = i[\hat H, \hat
r_\alpha]/\hbar$. In a non-orthogonal basis it needs the position matrix
$\langle i|\hat r_\alpha|j\rangle$ (the dipole integrals) as well as $H$ and
$S$:

$$
v_\alpha = \frac{i}{\hbar}\,S^{-1}\big(H S^{-1} r_\alpha - r_\alpha S^{-1} H\big).
$$

An `lcao` model computes the dipole integrals from its numerical orbitals, so
the velocity is available for every `lcao` model. A non-orthogonal `sktb` or
`dftb` model has no orbital functions to integrate, so it provides neither
position nor velocity, and the transport tasks are not available for it.

A non-orthogonal `tb_file` model has no orbital functions either, but can choose
`"position": "mulliken"`, $X_{ij}(\mathbf R) = \tfrac12(\mathbf r_i + \mathbf R
+ \mathbf r_j) S_{ij}(\mathbf R)$. In the eigenbasis this differs from the true
velocity by a term proportional to $E_n - E_m$, so band velocities are exact and
the error grows with the transition energy.

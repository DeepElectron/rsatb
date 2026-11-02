---
template: home.html
title: RSATB
description: >-
  RSATB computes electronic observables of large tight-binding and LCAO
  systems — DOS, LDOS, density, conductivity, diffusion — with random-state
  Chebyshev methods, distributed over MPI and OpenMP.
hide:
  - navigation
  - toc
---

# RSATB

<p class="rs-section-kicker">What it is</p>

## One JSON file, one command, any system size

Linear-scaling DFT and machine-learned electronic-structure models now produce
Hamiltonians with millions of orbitals. Extracting physics from them is the
bottleneck: dense diagonalization is out of reach and most tools assume an
orthogonal basis. RSATB replaces the eigensolve with **random-state trace
estimators** and **Chebyshev recurrences** on a distributed sparse matrix, and
treats a non-orthogonal overlap through a graph-local $S^{-1/2}$ factor.

<div class="rs-pipeline">
  <div><b>structure</b><span>where the atoms are — POSCAR, CIF, XYZ or inline</span></div>
  <div><b>model</b><span>how H(R) and S(R) are built — SK, DFTB, LCAO, file</span></div>
  <div><b>run</b><span>k-points, random samples, spectral window, preconditioner</span></div>
  <div><b>tasks</b><span>the observables, computed in order against one setup</span></div>
  <div><b>output</b><span>one <code>.npz</code> per task and a <code>summary.json</code></span></div>
</div>

<div class="grid cards" markdown>

-   :material-sine-wave:{ .lg } **No dense diagonalization**

    ---

    Sparse matrix–vector products replace the global eigensolve, so cost and
    memory grow linearly with the number of atoms.

-   :material-vector-triangle:{ .lg } **Non-orthogonal by construction**

    ---

    LCAO and DFTB overlaps are handled by a reusable, graph-local
    $S^{-1/2}$ preconditioner — no orthogonalized basis required.

-   :material-server-network:{ .lg } **Built for clusters**

    ---

    k-points, atoms and block rows are distributed across MPI ranks and
    OpenMP threads. The same input runs on a laptop and on 64 nodes.

-   :material-shield-check-outline:{ .lg } **Inputs that check themselves**

    ---

    A closed schema rejects typos with a suggestion. `rsatb check` validates
    structure, files and physics consistency before anything is queued.

</div>

<p class="rs-section-kicker">Observables</p>

## Twelve tasks, one setup

| Goal | Tasks |
|---|---|
| Spectrum and populations | [`energy_bounds`](reference/input/tasks/energy_bounds.md), [`dos`](reference/input/tasks/dos.md), [`pdos`](reference/input/tasks/pdos.md), [`occupation`](reference/input/tasks/occupation.md), [`fermi_level`](reference/input/tasks/fermi_level.md) |
| Real-space electronic structure | [`density`](reference/input/tasks/density.md), [`ldos`](reference/input/tasks/ldos.md) |
| Electrical and optical response | [`dc`](reference/input/tasks/dc.md), [`dc_spectrum`](reference/input/tasks/dc_spectrum.md), [`ac`](reference/input/tasks/ac.md) |
| Dynamics and quantum geometry | [`msd`](reference/input/tasks/msd.md), [`quantum_metric`](reference/input/tasks/quantum_metric.md) |

Tasks run in order against a single setup: a DOS and an optical spectrum in
the same input share the Hamiltonian, the overlap treatment and the domain
decomposition.

<p class="rs-section-kicker">Results</p>

## Validated, and scalable

<figure markdown>
![Validation of RSATB against exact diagonalization](assets/figures/fig_validation.png)
<figcaption><strong>Validated across materials and observables.</strong> RSATB
against exact diagonalization for diamond, silicon and aluminium: DOS, charge
density, LDOS, electrical and optical conductivity and the diffusion
coefficient.</figcaption>
</figure>

<figure markdown>
![CPU, memory and wall-time scaling to one million atoms](assets/figures/fig_scaling.png)
<figcaption><strong>Near-linear scaling to 1,004,760 atoms.</strong> CPU time and
memory grow linearly with system size; wall time stays nearly constant under
weak scaling across nodes.</figcaption>
</figure>

<figure markdown>
![Twisted trilayer graphene supermoiré application](assets/figures/fig_application.png)
<figcaption><strong>A 5.92-million-orbital supermoiré.</strong> Spectra and
spatially resolved electronic structure of a 455,140-atom twisted trilayer
graphene supermoiré.</figcaption>
</figure>

<p class="rs-section-kicker">Start here</p>

## Where to go next

<div class="grid cards" markdown>

-   :material-download:{ .lg } **[Install](getting-started/installation.md)**

    ---

    A conda environment with MPI, HDF5 and the native solver, in one command.

-   :material-rocket-launch-outline:{ .lg } **[Quick start](getting-started/quickstart.md)**

    ---

    Graphene from a template to a density of states in under a minute.

-   :material-school-outline:{ .lg } **[Tutorials](tutorials/index.md)**

    ---

    Seven worked calculations, from Slater-Koster graphene to non-orthogonal
    LCAO optics and real-space maps.

-   :material-book-open-variant:{ .lg } **[Input reference](reference/input/index.md)**

    ---

    Every key, type, unit and default — generated from the schema the
    validator uses.

</div>

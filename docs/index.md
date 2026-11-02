---
template: home.html
title: RSATB
description: >-
  RSATB computes densities of states, conductivities, optical spectra and
  real-space maps of million-atom tight-binding and DFT Hamiltonians, from a
  short JSON input, on a laptop or a cluster.
hide:
  - navigation
  - toc
  - footer
---

# RSATB

<p class="rs-section-kicker">How it works</p>

## Three steps, any system size

<p class="rs-section-sub">Describe the calculation in a short JSON file, check
it, and run it. The same file works on a laptop and on hundreds of cores.</p>

<div class="rs-steps">
  <div class="rs-step">
    <span class="rs-step__n">1</span>
    <h3>Describe</h3>
    <p>Say where the atoms are, which Hamiltonian to use, and what to compute.
    Start from a template.</p>
    <code class="rs-cmd">rsatb template sktb -o run.json</code>
  </div>
  <div class="rs-step">
    <span class="rs-step__n">2</span>
    <h3>Check</h3>
    <p>Catch typos, missing files and impossible requests in seconds — before
    anything is queued.</p>
    <code class="rs-cmd">rsatb check run.json</code>
  </div>
  <div class="rs-step">
    <span class="rs-step__n">3</span>
    <h3>Run</h3>
    <p>Results arrive as NumPy files plus a summary that records exactly what
    was computed.</p>
    <code class="rs-cmd">mpirun -n 64 rsatb run run.json</code>
  </div>
</div>

<p class="rs-section-kicker">Observables</p>

## What you can compute

<p class="rs-section-sub">Twelve observables, from a first density of states
to optical spectra. Ask for several in one run — they share the setup.</p>

<div class="grid cards rs-grid-3" markdown>

-   :material-chart-bell-curve:{ .lg } **[Density of states](tutorials/graphene-dos.md)**

    Total and projected, at any resolution, for systems of millions of atoms.

-   :material-flash-outline:{ .lg } **[Conductivity](tutorials/transport.md)**

    DC conductivity across the whole spectrum, or at a chosen Fermi level and
    temperature.

-   :material-white-balance-sunny:{ .lg } **[Optical spectra](tutorials/optics.md)**

    Optical conductivity, dielectric function, refractive index and
    absorption.

-   :material-run-fast:{ .lg } **[Diffusion](tutorials/transport.md)**

    How fast electrons spread at each energy — ballistic, diffusive or
    localized.

-   :material-cube-scan:{ .lg } **[Real-space maps](tutorials/realspace.md)**

    Electron density and local density of states on a 3D grid.

-   :material-scale-balance:{ .lg } **[Fermi level & occupation](reference/input/tasks/fermi_level.md)**

    Chemical potential, electron count and band energy at any temperature.

</div>

<p class="rs-section-kicker">Models</p>

## Bring your own Hamiltonian

<p class="rs-section-sub">Write a model by hand, or load one from the code
you already use. Overlapping atomic orbitals are handled natively.</p>

<div class="grid cards rs-grid-4" markdown>

-   :material-pencil-ruler:{ .lg } **Slater-Koster**

    Hopping parameters you write directly in the input.

-   :material-atom:{ .lg } **DFT**

    ABACUS, DeePTB and NextHam Hamiltonians with their atomic orbitals.

-   :material-table-large:{ .lg } **DFTB**

    Standard `.skf` parameter sets such as mio and 3ob.

-   :material-file-chart-outline:{ .lg } **Tight-binding files**

    Any ready-made orthogonal $H(\mathbf R)$, e.g. a DeePTB export.

</div>

<p class="rs-section-kicker">Results</p>

## Accurate, and built to scale

<p class="rs-section-sub">Checked against exact diagonalization, and run on
systems of up to a million atoms.</p>

=== "Accuracy"

    <figure markdown>
    ![Validation of RSATB against exact diagonalization](assets/figures/fig_validation.png)
    <figcaption><strong>Matches exact results</strong> for diamond, silicon
    and aluminium across six observables: density of states, charge density,
    local density of states, electrical and optical conductivity, and
    diffusion.</figcaption>
    </figure>

=== "Scaling"

    <figure markdown>
    ![CPU, memory and wall-time scaling to one million atoms](assets/figures/fig_scaling.png)
    <figcaption><strong>Near-linear scaling to 1,004,760 atoms.</strong> Cost
    and memory grow in proportion to system size; adding nodes with the
    system keeps the wall time nearly constant.</figcaption>
    </figure>

=== "Application"

    <figure markdown>
    ![Twisted trilayer graphene supermoiré application](assets/figures/fig_application.png)
    <figcaption><strong>A 455,140-atom twisted trilayer graphene
    supermoiré</strong> with 5.92 million orbitals: spectra and real-space
    electronic structure from one input.</figcaption>
    </figure>

<p class="rs-section-kicker">Documentation</p>

## Find your way

<div class="grid cards rs-grid-4" markdown>

-   :material-rocket-launch-outline:{ .lg } **[Get started](getting-started/index.md)**

    Install RSATB and run your first calculation in a few minutes.

-   :material-school-outline:{ .lg } **[Tutorials](tutorials/index.md)**

    Seven worked examples, from beginner to advanced, about two minutes each.

-   :material-compass-outline:{ .lg } **[Guides](guides/index.md)**

    Short answers to specific tasks: importing data, convergence, clusters.

-   :material-book-open-variant:{ .lg } **[Reference](reference/index.md)**

    Every input option, output file and command, generated from the source.

</div>

<div class="rs-cta">
  <div>
    <h2>Ready to try it?</h2>
    <p>Your first density of states takes one template and one command.</p>
  </div>
  <a class="rs-btn rs-btn--primary" href="getting-started/quickstart/">Start the quick start
    <svg viewBox="0 0 24 24"><path d="M13.2 5.3 19.9 12l-6.7 6.7-1.4-1.4 4.3-4.3H4v-2h12.1l-4.3-4.3z"/></svg></a>
</div>

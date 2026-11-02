---
description: A learning path of seven worked RSATB calculations, from a first density of states to real-space maps and Python scripting. Each runs in about two minutes.
---

<p class="rs-kicker">Tutorials</p>

# Learn RSATB by example

<div class="rs-lead" markdown>
Seven short, worked calculations. Each one is a ready-to-run input from the
repository, explained step by step, with the result it actually produced.
They are ordered so that every tutorial adds one new idea.
</div>

## The basics

Start here. These three cover everything you need for a density of states
with any kind of Hamiltonian.

<div class="grid cards rs-grid-3" markdown>

-   :material-numeric-1:{ .lg } **[Graphene density of states](graphene-dos.md)**

    Write a model by hand, run it, and check it against the exact answer.

    <span class="rs-chip rs-chip--beginner">Beginner</span> <span class="rs-chip">≈ 2 min</span>

-   :material-numeric-2:{ .lg } **[A DFT Hamiltonian](lcao-diamond.md)**

    Load diamond from a DFT calculation and its atomic orbitals.

    <span class="rs-chip rs-chip--beginner">Beginner</span> <span class="rs-chip">&lt; 1 min</span>

-   :material-numeric-3:{ .lg } **[Other model types](other-models.md)**

    DFTB parameter files and ready-made tight-binding files.

    <span class="rs-chip rs-chip--beginner">Beginner</span> <span class="rs-chip">&lt; 1 min</span>

</div>

## Observables

Beyond the density of states: how electrons move, absorb light, and where
they sit in space.

<div class="grid cards rs-grid-3" markdown>

-   :material-numeric-4:{ .lg } **[Transport](transport.md)**

    Conductivity and diffusion in graphene, checked against theory.

    <span class="rs-chip rs-chip--intermediate">Intermediate</span> <span class="rs-chip">≈ 2 min</span>

-   :material-numeric-5:{ .lg } **[Optical spectra](optics.md)**

    The optical conductivity and dielectric function of diamond.

    <span class="rs-chip rs-chip--intermediate">Intermediate</span> <span class="rs-chip">≈ 2 min</span>

-   :material-numeric-6:{ .lg } **[Real-space maps](realspace.md)**

    Electron density and local density of states on a 3D grid.

    <span class="rs-chip rs-chip--intermediate">Intermediate</span> <span class="rs-chip">&lt; 1 min</span>

</div>

## Going further

<div class="grid cards rs-grid-2" markdown>

-   :material-numeric-7:{ .lg } **[Python scripting](python.md)**

    Sweep a parameter in a loop — something a single input file can't do.

    <span class="rs-chip rs-chip--advanced">Advanced</span> <span class="rs-chip">≈ 2 min</span>

</div>

## Before you start

You need RSATB [installed](../getting-started/installation.md) and a copy of
the repository, which holds the inputs and their data:

```bash
git clone --recurse-submodules https://github.com/DeepElectron/rsatb.git
cd rsatb
```

!!! tip "Running times"
    Times are for one process on a shared workstation. Your machine will
    usually be faster.

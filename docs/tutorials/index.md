---
description: A learning path of seven worked RSATB calculations, from a first density of states to real-space maps and Python scripting. Each runs in about two minutes.
---

<p class="rs-kicker">Tutorials</p>

# Learn RSATB by example

<div class="rs-lead" markdown>
Seven worked examples. Each uses an input file from the repository and
shows the output it produced. Later tutorials build on concepts introduced in
earlier ones.
</div>

## The basics

The density of states with each type of model.

<div class="grid cards rs-grid-3" markdown>

-   :material-numeric-1:{ .lg } **[Graphene density of states](graphene-dos.md)**

    A Slater-Koster model, compared with the exact result.

    <span class="rs-chip rs-chip--beginner">Beginner</span> <span class="rs-chip">≈ 2 min</span>

-   :material-numeric-2:{ .lg } **[A DFT Hamiltonian](lcao-diamond.md)**

    Diamond from a DFT Hamiltonian and its atomic orbitals.

    <span class="rs-chip rs-chip--beginner">Beginner</span> <span class="rs-chip">&lt; 1 min</span>

-   :material-numeric-3:{ .lg } **[Other model types](other-models.md)**

    DFTB parameter files and ready-made tight-binding files.

    <span class="rs-chip rs-chip--beginner">Beginner</span> <span class="rs-chip">&lt; 1 min</span>

</div>

## Observables

Conductivity and diffusion, optical response, and real-space maps.

<div class="grid cards rs-grid-3" markdown>

-   :material-numeric-4:{ .lg } **[Transport](transport.md)**

    Conductivity and diffusion in graphene.

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

    A parameter sweep with the Python API.

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
    Times are wall times for one process on a shared workstation.

---
description: Eight worked RSATB calculations, from a first density of states to real-space maps, ABACUS and NextHam Hamiltonians, and Python scripting.
---

<p class="rs-kicker">Tutorials</p>

# Learn RSATB by example

<div class="rs-lead" markdown>
Eight worked examples. Each tutorial shows its input file and the output
it produced.
</div>

## The basics

Tutorials 1 to 3 compute the density of states with each type of model.

<div class="grid cards rs-grid-3" markdown>

-   :material-numeric-1:{ .lg } **[Graphene density of states](graphene-dos.md)**

    Graphene with a Slater-Koster model, compared with the exact result.

    <span class="rs-chip rs-chip--beginner">Beginner</span>

-   :material-numeric-2:{ .lg } **[A DFT Hamiltonian](lcao-diamond.md)**

    Diamond, from a DFT Hamiltonian and its atomic orbital files.

    <span class="rs-chip rs-chip--beginner">Beginner</span>

-   :material-numeric-3:{ .lg } **[Other model types](other-models.md)**

    Models read from DFTB parameter files and from tight-binding files.

    <span class="rs-chip rs-chip--beginner">Beginner</span>

</div>

## Observables

Tutorials 4 to 6 compute transport, optical and real-space quantities.

<div class="grid cards rs-grid-3" markdown>

-   :material-numeric-4:{ .lg } **[Transport](transport.md)**

    The conductivity and diffusion of graphene.

    <span class="rs-chip rs-chip--intermediate">Intermediate</span>

-   :material-numeric-5:{ .lg } **[Optical spectra](optics.md)**

    The optical conductivity and dielectric function of diamond.

    <span class="rs-chip rs-chip--intermediate">Intermediate</span>

-   :material-numeric-6:{ .lg } **[Real-space maps](realspace.md)**

    The electron density and local density of states of diamond on a 3D
    grid.

    <span class="rs-chip rs-chip--intermediate">Intermediate</span>

</div>

## Going further

<div class="grid cards rs-grid-2" markdown>

-   :material-numeric-7:{ .lg } **[ABACUS and NextHam](abacus-nextham.md)**

    Hamiltonians from ABACUS and NextHam, checked against ABACUS, for a
    metal and for a semiconductor with spin-orbit coupling.

    <span class="rs-chip rs-chip--advanced">Advanced</span>

-   :material-numeric-8:{ .lg } **[Python scripting](python.md)**

    A strain sweep of graphene with the Python API.

    <span class="rs-chip rs-chip--advanced">Advanced</span>

</div>

## Before you start

You need RSATB [installed](../getting-started/installation.md) and a copy of
the repository, which holds the inputs and their data:

```bash
git clone --recurse-submodules https://github.com/DeepElectron/rsatb.git
cd rsatb
```


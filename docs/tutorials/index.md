---
description: Seven worked RSATB calculations, each a runnable input from the repository, from Slater-Koster graphene to non-orthogonal LCAO optics, transport and real-space maps.
---

<p class="rs-kicker">Tutorials</p>

# Tutorials

<div class="rs-lead" markdown>
Each tutorial is one runnable input from the repository's `examples/`
directory, explained line by line, with the output and figure it actually
produced. They build on each other, but each stands alone.
</div>

<div class="grid cards" markdown>

-   **1 · [Graphene density of states](graphene-dos.md)**

    ---

    A Slater-Koster model written inline, tiled to 32,768 atoms, checked
    against the exact answer. *Concepts: supercells, `dE`, `nsamples`,
    labels.*

-   **2 · [A non-orthogonal LCAO model](lcao-diamond.md)**

    ---

    Diamond from a DFT Hamiltonian and its numerical orbitals. *Concepts:
    overlap, $S^{-1/2}$, the spectral window, k-point paths.*

-   **3 · [Optical response](optics.md)**

    ---

    DOS and AC conductivity of a 128-atom diamond cell from one setup.
    *Concepts: Fermi level, dielectric function, absorption.*

-   **4 · [Transport and diffusion](transport.md)**

    ---

    DC conductivity and wave-packet spreading in graphene. *Concepts:
    velocity operator, 2D normalization, ballistic transport.*

-   **5 · [Real-space maps](realspace.md)**

    ---

    Electron density and LDOS on a grid, written in parallel and reassembled.
    *Concepts: Gamma-only tasks, distributed output.*

-   **6 · [DFTB and file Hamiltonians](other-models.md)**

    ---

    The remaining two model families: `.skf` tables and an orthogonal H(R)
    file.

-   **7 · [Scripting with Python](python.md)**

    ---

    A strain sweep that no single input can express, through the Python API.

</div>

## Before you start

[Install RSATB](../getting-started/installation.md) and clone the repository —
the inputs and their data live in it:

```bash
git clone --recurse-submodules https://github.com/floatingCatty/rsatb.git
cd rsatb
```

Every input resolves its paths against its own location, so you can run them
from anywhere. Results are written next to the input, under `results/`.

!!! info "Timings"
    Wall times quoted in the tutorials were measured on one rank with 8
    OpenMP threads on a shared, heavily loaded node. Expect yours to be
    faster.

---
description: Tutorial — the dftb model from a directory of .skf files, and the tb_file model for an orthogonal H(R) such as a DeePTB Slater-Koster export.
---

<p class="rs-kicker">Tutorial 6</p>

# DFTB and file Hamiltonians

<div class="rs-lead" markdown>
The last two model families. `dftb` reads Slater-Koster tables from `.skf`
files; `tb_file` reads an orthogonal $H(\mathbf R)$ that some other program
already built. Both run every spectral task; what they can and cannot do
beyond that follows from their operators.
</div>

## DFTB parameter sets

<!-- rsatb-input: examples/03_dftb_skf.json -->

- **`skf_dir`** holds one file per *ordered* species pair: `C-C.skf`,
  `C-H.skf`, `H-C.skf`, …. The example ships a small synthetic `C-C.skf` so it
  runs with no downloads; for real work, point it at a released set such as
  mio or 3ob.
- **`basis` is declared.** A `.skf` file tabulates all ten channels up to $d$
  whether the parameterization uses them or not, so the file cannot say which
  shells belong to the basis. DFTB+ takes the same information from its input.
- **`heteronuclear`** decides what to do with a mixed-shell channel whose
  reading the declared bases do not force; the default refuses rather than
  guesses.
- **`run.kpoints`** is a Gamma-centred $2\times2\times2$ mesh; the DOS comes
  back with eight rows, in mesh order.

```bash
rsatb run examples/03_dftb_skf.json
```

```text
  model         dftb  C
  capabilities  non-orthogonal
  k-points      8 (2x2x2 mesh)
  ...
  basis         C · 4 orbitals · cutoff 1.500 Å
  bounds        [-22.268290, -1.472636] eV
```

`capabilities: non-orthogonal` and nothing else: DFTB data always carries an
overlap but no dipole integrals, so there is no velocity or position operator
and the transport tasks are unavailable. `rsatb check` says so if you ask for
one.

## An orthogonal H(R) from a file

<!-- rsatb-input: examples/04_tb_file_chain.json -->

`tb_file` is the one model that must be told its basis, because its file
formats carry no basis header: `type_symbols`, `type_norb` (spin-expanded) and
`cutoff_radius` per species. The cutoff decides the graph — an edge exists
when $d \le r_i + r_j$ — so 1.1 Å per atom reaches the neighbour at 2.0 Å but
not the one at 4.0 Å.

The data is a 1D chain with $\varepsilon_0 = +0.5$ eV and $t = -1$ eV, whose
band is $E(k) = \varepsilon_0 + 2t\cos(ka)$:

```bash
rsatb run examples/04_tb_file_chain.json
```

```text
  model         tb_file
  capabilities  orthogonal, velocity, position
  k-points      11 (line through 2 vertices)
  ...
  bounds        [-2.000000, 3.000000] eV
```

The band runs from $\varepsilon_0 + 2t = -1.5$ eV to $\varepsilon_0 - 2t = 2.5$
eV; the window adds the 0.5 eV pad on both sides. Because the basis is
orthogonal, the position operator comes from the coordinates and the velocity
from the commutator, so a `tb_file` model runs every transport task.

## Which to use

| | `dftb` | `tb_file` |
|---|---|---|
| Source | DFTB parameter sets | DeePTB SK exports, any orthogonal TB |
| Overlap | yes | no |
| Basis | declared in `basis` | declared as `type_norb` + `cutoff_radius` |
| Transport tasks | no | yes |

The full comparison of all four families is in
[Choose a model](../guides/choose-model.md).

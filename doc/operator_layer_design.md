# Operator layer: a structure for the model differences

Scope: how a `Model` becomes the operators an observable runs against — the
path from `model/model.hpp` through `model/operator_construction.hpp` and
`model/operator_set.hpp`. The physics in those files is right, and none of it
changes here. What changes is where it lives.

The model layer got its structure in the setup/IO redesign: four families, four
files, one interface. The operator layer did not follow. It still has one
god-class, two interleaved namespaces, and the family differences spread across
four files — so a reader who knows exactly which family they care about still
cannot find the code that makes it different.

---

## 1. What is wrong

### 1.1 `OperatorConstruction` is three unrelated things in one class

[operator_construction.hpp](../rsatb/model/operator_construction.hpp) is 873
lines holding:

| lines | what it is | who needs it |
|---|---|---|
| `init`, `orbital_radials`, `has_beta_projectors`, `orbital_cutoffs`, `projector_cutoffs`, `velocity_graph`, `build_overlap`, `build_position`, `build_overlap_velocity` | the **op2c two-centre integral engine** | LCAO only |
| `build_from_reader` | **Hamiltonian file → H(R) images** | LCAO *and* `tb_file` |
| `set_local_block` ×4, `set_global_block`, `cal_phase` | **vbcsr block-assembly glue** | whoever assembles a block |
| `FirstLatticeMoment*`, `make_orthogonal_diagonal_coordinate_vector` | **generic operators from H(R) and geometry** | MSD, orthogonal position |

The consequences are visible at the call sites.
[tb_file_model.hpp:20](../rsatb/model/tb_file_model.hpp:20) includes the whole
header — and with it op2c — to reach one function that has nothing to do with
op2c. And `LcaoModel` names its member `integrals_`
([lcao_model.hpp:230](../rsatb/model/lcao_model.hpp:230)), which is the honest
name for the op2c engine and the wrong name for the class it points at.

### 1.2 The family differences are scattered over four files and two namespaces

"How does LCAO's velocity differ from SKTB's?" is the question the structure
should answer immediately. Today it takes four hops:

1. `lcao_model.hpp:120` — delegates to `integrals_->build_overlap_velocity(...)`
2. `operator_construction.hpp:556` — the op2c call, 200 lines
3. `sktb_model.hpp` — no `BuildVelocity` worth reading; it returns
   `ProvidesOperatorBuilders() == false`
4. `operator_set.hpp:395` — `OrthogonalImageVelocityBuilder`, where SKTB's
   velocity actually is

Nothing sits next to its alternative. The two implementations of the same
operator are in different files, in different namespaces, and one of them is
named after a *builder* rather than after the physics.

### 1.3 Generic physics is buried inside the container that caches it

`OrthogonalImageVelocityBuilder::build` is ~120 lines of
**v = (i/ħ)[H, r]** written inline in a class body, inside
[operator_set.hpp](../rsatb/model/operator_set.hpp) — the header whose job is
per-k-point *state*. It is the most reusable piece of operator construction in
the tree and the hardest to find. It cannot be tested without building an
`OperatorSet`, and an `OperatorSet` needs k-points, a preconditioner and MPI.

### 1.4 The namespaces interleave, exactly as §1.8 said they must not

```
model/lcao_model.hpp        (rsatb::model)   includes
  → model/operator_construction.hpp          (rsatb::backend)
model/operator_set.hpp      (rsatb::backend) includes
  → model/model.hpp                          (rsatb::model)
  → model/sktb/sktb_construction.hpp         (rsatb::model::sktb)
```

`rsatb::backend` and `rsatb::model` include each other, and both live under
`model/`. The setup/IO redesign resolved this for the Slater-Koster types; the
operator layer kept it.

### 1.5 `ProvidesOperatorBuilders()` is a strategy switch wearing a boolean

One `bool` on the `Model` interface selects between two entire implementations
of velocity *and* position:

```cpp
artifacts.velocity_builder = from_model
    ? std::make_unique<ModelVelocityBuilder>(&model)
    : std::make_unique<OrthogonalImageVelocityBuilder>(images.get());
```

Two problems. First, the name says nothing about which operators or which
route. Second, it is stated *independently of* `capabilities()`, which it is
actually tied to: the generic route derives v from H(R) alone, which is the
velocity only in an orthogonal basis. A model that returns
`capabilities().overlap == true` and `ProvidesOperatorBuilders() == false` is
self-contradictory, and nothing checks. The redesign removed
`install_default_fallbacks` because it *inferred* capabilities; the other half
— two declarations that must agree, with no cross-check — is still here.

### 1.6 The position triplet carries two representations at once

`PositionAxisData` documents the invariant
([operators.hpp:36](../rsatb/model/operators.hpp:36)):

> An orthogonal basis represents position by its diagonal alone, while a
> non-orthogonal basis needs the explicit matrix. **Exactly one of the two is
> populated for a given model.**

It is not kept. `position_state()`
([operator_set.hpp:193](../rsatb/model/operator_set.hpp:193)) calls
`attach_orthogonal_diagonal_position` on *every* triplet, including LCAO's,
which already has explicit matrices — so both are populated and the consumer
picks by re-deriving orthogonality from `has_overlap()`. It is benign today
only because every consumer happens to branch the same way. The construction is
also split in two: `OrthogonalDiagonalPositionBuilder::build` returns an
**empty** triplet and the `OperatorSet` fills it afterwards.

### 1.7 Four entry points where one would do

`BuildModelArtifacts` → `ModelArtifacts` → `finalize_operator_set` →
`build_operator_set`. Three of the four are intermediate, and `ModelArtifacts`
exists only to carry four members across one function boundary.

---

## 2. The design

One rule, which is the model layer's rule extended to operators:

> **Every operator has exactly one named source, and the source is declared
> where the model is defined — not chosen where the operators are stored.**

Three layers, strictly one-way:

```
  operator/   what an operator is, and how to build one from H(R) + geometry
     ↑            generic; knows nothing about any family
  model/      what THIS family's operators are, and where each comes from
     ↑            one directory per family
  setup/      assembling, caching and preconditioning them
```

### 2.0 What decides which directory something goes in

The axis is **who can use it**, not what it produces. That is worth stating
because `operator/` sounds like "everything that builds an operator", and it is
not:

| | belongs in |
|---|---|
| needs no family — works from H(R), the geometry, or a vbcsr matrix | `operator/` |
| is one family's own machinery | `model/<family>/` |
| is shared by two families because they genuinely do the same thing | `model/` |
| holds per-run state: k-points, preconditioner, bounds | `setup/` |

So the op2c two-centre engine stays in `model/lcao/`, even though its whole job
is building operators. Only LCAO can use it. Moving it to `operator/` would put
op2c back on every model's include path -- §1.1's defect, where `tb_file` pulled
in op2c to reach one unrelated function -- and would move LCAO's operator
construction away from LCAO, which is §1.2's.

The reverse test: the commutator velocity lives in `operator/` not because it is
"generic-sounding" but because SKTB, `tb_file`, and any future orthogonal family
all use the same one.

### 2.1 Layout

```
rsatb/operator/                       namespace rsatb::op
  block_assembly.hpp    vbcsr glue: set_local_block / set_global_block /
                        cal_phase, as free functions. No state, no op2c.
  commutator_velocity.hpp   v = (i/hbar) [H, r] from H(R) and positions.
  diagonal_position.hpp     r as a diagonal, in an orthogonal basis.
  lattice_moment.hpp        the first lattice moment, for MSD.
  builders.hpp              IVelocityBuilder / IPositionBuilder, plus the
                            generic implementations of each.

rsatb/model/                          namespace rsatb::model
  model.hpp             the interface: basis, spin, operator sources, builders
  operator_source.hpp   NEW -- OperatorSource / OperatorSources
  basis_spec.hpp, operators.hpp        (unchanged)
  hamiltonian_images.hpp  file -> H(R). Shared by LCAO and tb_file; no op2c.
  lcao/
    lcao_model.hpp
    op2c_engine.hpp     the op2c integral engine, named for what it is
  tbfile/
    tb_file_model.hpp
  sktb/
    sktb_model.hpp, sktb_construction.hpp, sk_table.hpp, ...   (unchanged)

rsatb/setup/                          namespace rsatb::setup
  operator_set.hpp      OperatorSet ONLY: k-point state, lazy caches,
                        accessors. No construction, no builder classes.
  operator_assembly.hpp BuildOperatorSet(model, atomic_data, options, runtime)
                        -- the one place a Model becomes an OperatorSet.
  kpoint_partition.hpp, spectral_bounds.hpp, dense_bounds.hpp   (unchanged)
```

`rsatb::backend` goes away. 37 references across 9 files become `op::` or
`setup::`.

### 2.2 The key move: declare the source, derive the capability

```cpp
// model/operator_source.hpp
namespace rsatb {
namespace model {

/// Where one operator comes from.
enum class OperatorSource {
    kUnavailable,  // this model cannot supply it; observables refuse by name
    kModel,        // the model's own integrals -- op2c for LCAO, SK tables for SKTB
    kGeneric,      // built from H(R) and the geometry alone
};

/// The source of every operator, stated once per family.
///
/// This replaces a `bool ProvidesOperatorBuilders()` that selected between two
/// whole implementations without naming either, and a `Capabilities` that had
/// to agree with it by hand.
struct OperatorSources {
    OperatorSource hamiltonian = OperatorSource::kModel;   // always the model
    OperatorSource overlap     = OperatorSource::kUnavailable;
    OperatorSource velocity    = OperatorSource::kUnavailable;
    OperatorSource position    = OperatorSource::kUnavailable;

    /// Capabilities are derived, so they cannot disagree with the sources.
    Capabilities capabilities() const;

    /// The one consistency rule, checked once instead of assumed everywhere:
    /// kGeneric derives the operator from H(R) alone, which is correct only in
    /// an orthogonal basis. kGeneric together with an overlap is a
    /// contradiction, and is rejected at model construction.
    void Validate(const std::string& family) const;
};

}  // namespace model
}  // namespace rsatb
```

`Model` then exposes `virtual OperatorSources operator_sources() const = 0;`
and `capabilities()` becomes non-virtual, derived from it.
`ProvidesOperatorBuilders()` is deleted.

### 2.3 The whole family difference, in one table

Each model states its own row, in its own file, in one place:

| family | H | S | v | r |
|---|---|---|---|---|
| **LCAO** | model — file reader | model — op2c | model — op2c *(needs `.upf`)* | model — op2c |
| **SKTB**, no overlap | model — SK tables | — | generic — commutator | generic — diagonal |
| **SKTB** with overlap | model — SK tables | model — SK tables | — | — |
| **DFTB** | model — SK tables | model — SK tables | — | — |
| **tb_file** | model — file reader | — | generic — commutator | generic — diagonal |

That table *is* the answer to "how do the families differ", and after this
change it is also the code. `rsatb doc` can print it, and `rsatb check`'s
capability cross-check ([check.py](../rsatb/inputs/check.py)) — which today
duplicates this knowledge in Python — can be generated from it rather than
maintained alongside it.

### 2.4 What each piece becomes

| today | after |
|---|---|
| `backend::OperatorConstruction` (873 lines, three jobs) | split three ways, below |
| ├ op2c engine | `model::lcao::Op2cEngine` in `model/lcao/op2c_engine.hpp` |
| ├ `build_from_reader` | `model::BuildHamiltonianImages(atomic_data, reader, comm)` in `model/hamiltonian_images.hpp` |
| ├ `set_*_block`, `cal_phase` | free functions in `operator/block_assembly.hpp` |
| └ `FirstLatticeMoment*` | `op::BuildLatticeMoment(...)` in `operator/lattice_moment.hpp` |
| `OrthogonalImageVelocityBuilder::build` (120 lines inline) | `op::BuildCommutatorVelocity(atomic_data, h_images, kpoint)` |
| `OrthogonalDiagonalPositionBuilder` + `OperatorSet::attach_orthogonal_diagonal_position` (two halves) | `op::BuildDiagonalPosition(atomic_data)`, one function, returns a complete triplet |
| `ModelArtifacts` + `BuildModelArtifacts` + `finalize_operator_set` + `build_operator_set` | one `setup::BuildOperatorSet(...)` |
| `Model::ProvidesOperatorBuilders()` | `Model::operator_sources()` |
| `Model::capabilities()` (virtual, hand-written per family) | non-virtual, derived from `operator_sources()` |
| `rsatb::backend` | `rsatb::op` (generic) + `rsatb::setup` (assembly) |

`OperatorSet` keeps only what its name promises: the local k-points, the
per-k-point state, the lazy caches, the bounds and the preconditioner. It goes
from 636 lines to about 320 -- lines 354-636 and 395-544 leave entirely --
and it stops being a construction site.

`operator/commutator_velocity.hpp` gains what the inline version could not
have: a docstring naming the formula and its precondition, and a test that
builds H(R) for a two-site chain and checks v against the analytic
∂E/∂k — with no `OperatorSet`, no k-point partition and no preconditioner.

### 2.5 Adding a family, after

1. `model/<family>/<family>_model.hpp` — implement `Model`, fill one
   `OperatorSources` row.
2. If it brings its own integrals, put the engine beside it in the same
   directory.
3. Nothing else. No new arm in an assembly function, no new builder class in
   the operator set, no new branch anywhere in `setup/`.

Today steps 1–3 also require editing `BuildModelArtifacts`, and knowing that
`ProvidesOperatorBuilders()` silently means "and my basis is non-orthogonal".

---

## 3. What landed

Implemented. The layout as built:

```
rsatb/operator/                       namespace rsatb::op          lines
  types.hpp                 VelocityTriplet, PositionTriplet, KPoint   61
  geometry.hpp              CellShiftComponent, image/graph checks     75
  block_assembly.hpp        BlochPhase, AddLocalBlock, AddGlobalBlock 120
  commutator_velocity.hpp   v = (i/hbar)[H, r]                        174
  diagonal_position.hpp     r as a diagonal, orthogonal bases          78
  lattice_moment.hpp        M_alpha = sum_R R_alpha A(R)               88
  builders.hpp              the two interfaces + the generic pair      87

rsatb/model/                          namespace rsatb::model
  model.hpp                 the interface                             125
  operator_source.hpp       OperatorSource / OperatorSources          107
  hamiltonian_images.hpp    file -> H(R); LCAO and tb_file            142
  lcao/op2c_engine.hpp      the op2c engine, LCAO only                563
  lcao/units.hpp            op2c's Bohr convention                     23
  lcao/lcao_model.hpp                                                 246
  tbfile/tb_file_model.hpp                                            114
  sktb/...                  unchanged

rsatb/setup/                          namespace rsatb::setup
  operator_set.hpp          the set, and nothing else       367  (was 636)
  operator_assembly.hpp     model -> operators, one place             164
```

`rsatb::backend` is gone. `model/operator_construction.hpp` (873 lines) and
`model/operator_set.hpp` (636) are gone; what was in them is above, and the
largest remaining file is the op2c engine, which is one thing.

### Where the design changed on contact

**Four entry points became two, not one.** `AssembleOperatorSet(parts, ...)`
survives alongside `BuildOperatorSet(model, ...)`, because assembling a set
from hand-made images and a stub builder is how `test_operator_set` tests the
caching, the bounds and the lazy construction without a model, a file or op2c.
That is a real seam; `OperatorAssembly` names it, and `ModelArtifacts` never
did. The three that vanished were the unnamed intermediates.

**Two headers the plan did not anticipate.** `operator/geometry.hpp` holds what
turned out to be shared by three constructions -- the Cartesian shift of an
image, whether an axis translates at all, and the "these images belong to this
graph" precondition. `model/lcao/units.hpp` holds the Bohr conversion, because
`calc/collocation.hpp` evaluates the same op2c radials on a real-space grid and
needs the factor without pulling in the engine.

**`model/operators.hpp` moved too.** It became `operator/types.hpp` in
namespace `op`. The operator value types are the operator layer's vocabulary;
leaving them in `model/` would have kept `setup/` depending on `model/` for a
type that has nothing to do with any family.

**Four `set_local_block` overloads became one template.** They differed only in
whether the shift arrived as `int` or `double` and the values as real or
complex. Both are template parameters now.

**A builder is a function, not a class hierarchy.** The first cut kept
`IVelocityBuilder`/`IPositionBuilder` with four implementations, two of which
(`ModelVelocityBuilder`, `ModelPositionBuilder`) held one pointer and forwarded
one call. All a builder ever needs to be is "something callable that outlives
assembly", and `std::function` is that. Six types became two aliases in
`operator/types.hpp` plus two factories in `operator/builders.hpp`; the model
route is a lambda written where it is chosen, so the call site names the
function it calls. `builders.hpp` went from 87 lines to 52, and the test's stub
builder became a lambda over a counter.

**Assembling from parts is `OperatorSet::FromParts`, a member.** The first cut
made it a free function, which needed `friend` -- and the friend declaration was
the *only* reason `operator_set.hpp` included `model/model.hpp`, so the free
function reintroduced the very dependency the split was meant to remove. As a
static member there is no friend and no include: "assemble a set from parts"
belongs to the set, and "read a model's source table" belongs to
`operator_assembly.hpp`. That is now the whole of why one depends on `model/`
and the other does not.

### Two defects fixed on the way

**The position invariant is kept now.** `PositionAxisData` documents "exactly
one of the two is populated"; the set used to attach the orthogonal diagonal to
every triplet, LCAO's included. It was benign only because every consumer
re-derived orthogonality before reading. `BuildDiagonalPosition` now returns a
complete triplet and nothing post-fills it, so an LCAO position has a matrix
and no diagonal. `test_operator_set` asserted the old behaviour and now asserts
the documented one.

**`calc/collocation.hpp` was relying on a leaked `using namespace vbcsr`** that
came in through `operator_construction.hpp`. Splitting that header surfaced it;
`dgemm_` is qualified now.

### Why the model does not own its operators

Asked directly, and worth recording. A `Model` is a *description* -- basis,
spin, SK tables or op2c tables. An `OperatorSet` is that description applied to
a **structure, a k-point set and an MPI communicator**. The relationship is
one-to-many, and not hypothetically:
[test_supercell_bonds.py](../test/test_supercell_bonds.py) builds one
`pz_model()` and four `RSATB`s from it, across two supercells and two ways of
writing the same basis atom. If the model owned the operators it could own one
instance, or would need a map keyed by (atomic data, k-points, communicator).

Two further consequences point the same way. Every model method is `const`
(`BuildHamiltonianImages(...) const`), and pybind keeps one model alive across
instances with `keep_alive`; owning mutable per-run state would break both. And
the set holds things that are not operators at all -- which ranks own which
k-points, S^-1/2, the spectral window. Those belong to a calculation, not to
"what the Hamiltonian is".

What the question did surface is that the set carried a capability flag it never
read: `OperatorCapabilities` was written by the assembler and never used, because
every `has_*` already answers from whether the artifact is present. The struct,
the member and the eight assignments are gone. Which operators exist is now
stated exactly once, by the model, and observed from the artifacts.

### Verification

No physics change: every moved body is the same code. The LCAO diamond
reference is bit-identical (`emin = -8.621686512892`), the graphene analytic
window is still `+-8.6` and the 1D chain `[-2.0, 3.0]`.

C++: `test_sktb`, `test_skf`, `test_operator_api` (14), `test_setup_stages`,
`test_operator_set` (7), `test_rsatb`, `test_op2c_engine`, `test_collocation`,
`test_hdf5_io`, `test_preconditioner` all pass. Python: 74 pass, plus
`test_binding.py` and `test_graphene_pz_dos.py`.

Two new tests close the last duplicated statement of the family table:
`test_capabilities_match_the_compiled_models` builds one model of each family
and asserts `check.py`'s pure-Python table equals the compiled
`capabilities()`, and `test_an_sktb_overlap_turns_off_velocity_in_both_places`
pins the orthogonality hinge in both. `check.py` still restates the table --
`rsatb check` has to run where the extension cannot -- but the restatement now
fails a test if it drifts.

---

## 4. The order it was done in

Each step compiled and passed the suite on its own; none changed physics.

1. `operator/block_assembly.hpp` — the vbcsr glue, four overloads to one template.
2. `operator/geometry.hpp`, `operator/lattice_moment.hpp`,
   `operator/diagonal_position.hpp` — the constructions that need no model.
   The diagonal position's two halves became one function here, which is the
   position-invariant fix.
3. `operator/commutator_velocity.hpp` — the 120 inline lines lifted out and
   documented; `operator/builders.hpp` reduced to adapters.
4. `model/hamiltonian_images.hpp` — the file reader split out, so `tb_file`
   stopped including op2c.
5. `model/lcao/op2c_engine.hpp` + `units.hpp` — what remained of
   `OperatorConstruction`, renamed for what it is, with `lcao_model.hpp` beside
   it and `tb_file_model.hpp` in its own directory.
6. `model/operator_source.hpp` — `OperatorSources`, `Capabilities` derived,
   `ProvidesOperatorBuilders` deleted, one row written per family.
7. `setup/operator_set.hpp` + `setup/operator_assembly.hpp` — the set moved and
   trimmed, the entry points collapsed, `rsatb::backend` retired.
8. `test_json_input.py` — the family table cross-checked against the compiled
   models rather than merely restated.

## 5. What this does not touch

`calc/` kernels, `vbcsr`, `op2c`, the JSON input schema, and every observable
signature. The only user-visible change is that a model asking for an operator
it cannot build now fails at construction rather than at first use, with the
family and the operator named.

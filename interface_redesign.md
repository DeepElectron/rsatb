# RSATB Setup / Initialization / IO Redesign

Scope: the user-facing path only — how a structure, a model, and run options get
turned into a working `RSATB` object. The `calc/` kernels, `vbcsr`, and `op2c`
are not touched.

Goal: three model families (**LCAO**, **SKTB**, **DFTB**), each with one coherent
story, reachable through one constructor, with no value the user has to
hand-transcribe and no config field that is silently ignored.

---

## 1. What is wrong today

### 1.1 The initialization is circular, so the model is passed twice

`RSATB`'s constructor needs `r_max_per_type` and `type_norb`. Both are
properties of the **model**, not of the structure. But the model is only
supplied later, in `setup()`. To break the cycle, the current code makes the
user run the model through a side channel first:

```python
model = SKTBModel.native(basis=..., onsite=..., pairs=...)
atomic_spec = model.prepare_atomic_spec_from_file(str(structure))  # round-trip into C++
rsatb = RSATB.from_file(npar, filename, atomic_spec.r_max_per_type, atomic_spec.type_norb)
rsatb.setup_sktb(model, SetupOptions(...))                         # model passed again
```

This is why `pybind_rsatb.cpp` carries two free functions
(`_prepare_sktb_atomic_spec_from_file`, `_prepare_sktb_atomic_spec_from_atomic_numbers`)
that exist for no reason other than to run half of the model resolution early.

For LCAO there is not even that much help — the user hand-transcribes the
numbers out of the basis-set file:

```python
RSATB.from_file(npar=5, filename="diamond_sc111.vasp", r_max_per_type=[3.7], type_norb=[13])
```

`3.7` is `C_gga_7au_100Ry_2s2p1d.orb`'s cutoff (7 a.u. = 3.70 Å) and `13` is
`2*1 + 2*3 + 1*5`. **op2c already knows both** — `Op2c::get_orb_rcut_max(itype)`
([op2c.hpp:247](rsatb/extern/op2c/src/int2c/op2c.hpp:247)) and
`RadialCollection::nzeta(itype, l)`. The user is re-deriving values the library
can compute, with no cross-check if they get them wrong.

### 1.2 Config crosses three layers as untyped dicts

Python dataclass → `to_dict()` → pybind `parse_*` → C++ struct. Every field is
written four times, and `pybind_rsatb.cpp:15-227` is ~210 lines of hand-rolled
`if (cfg.contains("x")) ... = py::cast<T>(cfg["x"])`.

Unknown keys are silently dropped, so a typo (`"r_cutt"`, `"kpoint"`) is not an
error — it is a default value.

On top of that, `driver.py` adds a *second*, undocumented normalization layer
that guesses the user's intent from dict shape: `_matrix_key`, `_split_onsite`,
`_split_pairs`, `_looks_like_matrix_group`, `_lower_pair_channels`,
`_pair_species`, `_radial_config`. `_build_sktb_native_config` accepts two
mutually exclusive input styles and raises if you mix them:

```python
if hamiltonian is not None and (onsite is not None or pairs is not None):
    raise ValueError("Use either legacy hamiltonian/overlap input or new onsite/pairs input, not both.")
```

Two input dialects for the same data, disambiguated by inspecting key names, is
the single biggest source of "hard to read and maintain" in the SKTB path.

### 1.3 The three model families are not actually three things

- **SKTB and DFTB are the same type.** `setup.hpp:74-75`:
  ```cpp
  using SKTBNativeInput = rsatb::backend::SKTBModel;
  using DFTBNativeInput = rsatb::backend::SKTBModel;   // identical
  ```
  `build_dftb_artifacts` differs from `build_sktb_artifacts` by one bool
  (`require_native_overlap`). **There is no `.skf` reader** — `DFTBModel` makes
  the user hand-build the same SK dict as SKTB. The DFTB file format the code
  claims to support does not exist yet.

- **LCAO's non-orthogonality is not enforced.** It is inferred from whether a
  string is empty: `LCAOIntegralInput::has_overlap_inputs()` is
  `!orb_names.empty() && !orb_dir.empty()`. Omit `orb_names` and you silently
  get an *orthogonal* calculation with an LCAO Hamiltonian, which is wrong
  physics, reported as success.

- **The h5 input is an axis, not a member.** All three configs carry
  `std::optional<Hdf5Input> input` and all three funnel into the same
  `build_hdf5_hamiltonian_images`. Meanwhile the h5 layout is hardcoded to the
  DeePTB convention — group `"0"`, dataset `"i_j_Rx_Ry_Rz"`
  ([hdf5.hpp:31-46](rsatb/io/hdf5.hpp:31)) — with **no format tag anywhere**, so
  a second producer's file cannot be added without editing the reader.

### 1.4 Spin is declared 2–3× and used ~once

Six spin fields are declared across two structs:

| field | `Hdf5Input` | `LCAOIntegralInput` | actually consumed? |
|---|---|---|---|
| `nspin` | ✓ | ✓ | only the `LCAOIntegralInput` copy, at [operator_set.hpp:506](rsatb/model/operator_set.hpp:506) |
| `lspinorb` | ✓ | ✓ | only the `LCAOIntegralInput` copy, at [operator_set.hpp:507](rsatb/model/operator_set.hpp:507) |
| `noncolin` | ✓ | ✓ | **never read anywhere** |

`build_hdf5_hamiltonian_images` uses `input.filename` and nothing else. So a
user who sets `nspin=2` on the Hamiltonian input gets it silently ignored, and
the two copies are free to disagree with no diagnostic.

### 1.5 `setup()` does six unrelated things and none are separately testable

One call chain — `RSATB::setup` → `build_operator_set` → `build_model_artifacts`
→ `finalize_operator_set` → `install_default_fallbacks` → `OperatorSet::initialize`
→ `sampling_.initialize` — performs file I/O, MPI k-point partitioning, H(k)/S(k)
sampling, preconditioner construction, **two PRIMME eigensolves per k-point**,
and random-vector allocation. There is no intermediate value to inspect or test.

`install_default_fallbacks` ([operator_set.hpp:866](rsatb/model/operator_set.hpp:866))
silently changes what physics is available based on whether `overlap_images` is
null — a velocity operator appears or does not, with no user-visible record.

### 1.6 `PreconditionerOptions::mode` is a string that frees memory

`mode` is stringly-typed over `{"H/S", "H", "S", "None"}`, case-sensitively
(`"none"` throws). `"H/S"` normalizes to `"S"`. And `"H"` has a side effect
nobody would guess from the name — it **deletes the overlap matrix**:

```cpp
if (mode == "H") { state.overlap.reset(); }   // operator_set.hpp:305
```

which then changes the return value of `has_overlap()` and makes some later
observables illegal. That is a memory optimization wearing a mode name.

### 1.7 Validation is duplicated and silently degrades

`RSATB::validate_pairwise_model_setup` re-runs `prepare_sktb_atomic_data(model)`
— recomputing exactly what Python already computed to build the `AtomicData` —
and compares it back against `atomic_data_->type_norb`. Two sources of truth for
one number, reconciled after the fact.

Worse, the cutoff check is skipped whenever `r_max_per_type_` is empty
([rsatb.hpp:797](rsatb/rsatb.hpp:797)), which is exactly the case for the
raw-array constructor (`r_max_per_type_.clear()` at
[rsatb.hpp:68](rsatb/rsatb.hpp:68)). The validation quietly does nothing on the
path most likely to need it.

### 1.8 Layering is circular

`backend/sktb/sktb.hpp` (namespace `rsatb::backend`)
← `model/setup.hpp` (namespace `rsatb::model`)
← `model/operator_set.hpp` (namespace `rsatb::backend` again).

The `model` and `backend` namespaces interleave rather than stack, so neither
can be read or compiled independently. `sktb.hpp` alone holds six concerns:
string trimming, the periodic table, orbital-label parsing, radial functions,
raw input structs, and resolved-model builders.

---

## 2. The redesign

### 2.1 Three concepts, one direction of dependency

| | owns | knows nothing about |
|---|---|---|
| **`Structure`** | atoms, cell, pbc | orbitals, cutoffs, models |
| **`Model`** | basis, spin, cutoffs, how to build H/S/v/r | MPI, k-points, observables |
| **`RunOptions`** | k-points, samples, preconditioner, bounds | which model family is in use |

```
io/  →  model/  →  setup/  →  calc/  →  rsatb.hpp
```

Strictly one-way. `model/` never includes `setup/`; the current
`backend ↔ model` cycle disappears.

### 2.2 The key move: the model derives the structure requirements

```cpp
// model/basis_spec.hpp
namespace rsatb::model {

// Everything AtomicData needs. Produced by the Model, never typed by the user.
struct BasisSpec {
  std::vector<std::string> type_symbols;   // species order; index == atom type
  std::vector<int> type_norb;              // orbitals per type, spin-expanded
  std::vector<double> cutoff_radius;       // per type; edge iff d <= r_i + r_j
};

struct SpinSpec {
  enum class Treatment { kNone, kCollinear, kNonCollinear, kSpinOrbit };
  Treatment treatment = Treatment::kNone;
  int Components() const;                  // basis-size multiplier: 1, 1, 2, 2
};

// What this model can produce. Declared, never inferred from empty strings.
struct Capabilities {
  bool overlap = false;                    // false => orthogonal basis
  bool velocity = false;
  bool position = false;
};

}  // namespace rsatb::model
```

This is what kills the circular initialization. `RSATB` asks the model for its
`BasisSpec` and builds the `AtomicData` itself:

```cpp
RSATB::RSATB(const Structure& structure, const Model& model,
             const RunOptions& options, const ParallelOptions& parallel);
```

One constructor. Model passed once. No `prepare_atomic_spec_*` round-trip, no
`_prepare_sktb_atomic_spec_from_*` pybind helpers, no hand-transcribed `3.7`.

### 2.3 The `Model` interface

```cpp
// model/model.hpp
class Model {
 public:
  virtual ~Model() = default;

  virtual std::string name() const = 0;
  virtual const BasisSpec& basis() const = 0;
  virtual const SpinSpec& spin() const = 0;
  virtual Capabilities capabilities() const = 0;

  // Real-space images. Built once, during construction.
  virtual std::unique_ptr<ImageContainer<double>> BuildHamiltonianImages(
      AtomicData& atomic_data) const = 0;

  // Returns nullptr if and only if capabilities().overlap is false.
  virtual std::unique_ptr<ImageContainer<double>> BuildOverlapImages(
      AtomicData& atomic_data) const = 0;

  // k-dependent operators. Built lazily, only when an observable asks.
  virtual std::unique_ptr<VelocityTriplet> BuildVelocity(
      AtomicData& atomic_data, const KPoint& k) const = 0;
  virtual std::unique_ptr<PositionTriplet> BuildPosition(
      AtomicData& atomic_data, const KPoint& k) const = 0;
};
```

`build_lcao_artifacts` / `build_sktb_artifacts` / `build_dftb_artifacts` /
`build_pairwise_tb_artifacts` / `install_default_fallbacks` / the `std::visit`
over `AnyModelConfig` all collapse into three subclasses. `ModelArtifacts` and
`AnyModelConfig` go away.

### 2.4 The three families, each with one story

#### LCAO — Hamiltonian file + orbitals + pseudopotentials, always non-orthogonal

```cpp
struct LcaoConfig {
  HamiltonianFile hamiltonian;        // format-tagged, see 2.5
  OrbitalFiles orbitals;              // REQUIRED — not optional
  PseudopotentialFiles pseudopotentials;   // optional; enables velocity
  SpinSpec spin;
  std::string log_file = "op2c.log";
};

class LcaoModel : public Model {
 public:
  // Fails if orbitals are missing. LCAO is non-orthogonal by definition,
  // so an LCAO model without an overlap source is a configuration error,
  // not a silently-orthogonal calculation.
  static absl::StatusOr<LcaoModel> Create(LcaoConfig config);

  Capabilities capabilities() const override {
    return {.overlap = true,                              // always
            .velocity = pseudopotentials_available_,
            .position = true};
  }
};
```

`basis()` is derived from the `.orb` files through op2c —
`type_norb[i] = Σ_l (2l+1)·nzeta(i,l) · spin.Components()`,
`cutoff_radius[i] = get_orb_rcut_max(i)` (plus `get_beta_rcut_max(i)` when
velocity is requested, matching op2c's own rule at
[op2c.cpp:386](rsatb/extern/op2c/src/int2c/op2c.cpp:386)).

Where the file also states a basis size, `Create` cross-checks it against the
derived one and fails on mismatch. Today nothing checks this.

#### SKTB — parameterized SK radial functions

```cpp
struct SktbConfig {
  std::map<std::string, std::vector<std::string>> basis;  // "C" -> {"pz"}
  SkParameters hamiltonian;                     // onsite + pair channels
  std::optional<SkParameters> overlap;          // absent => orthogonal
  SpinSpec spin;
};
```

One input dialect, not two. `SkParameters` is the resolved struct directly —
`{onsite: map<species, map<orbital, double>>, pairs: vector<SkPair>}` — with no
`_split_onsite` / `_looks_like_matrix_group` shape-guessing anywhere. H and S
are separate fields because they are separate physics, so there is nothing to
disambiguate.

`cutoff_radius` is derived from the radial functions exactly as
`derive_atomwise_cutoff_radii` does today; that logic is correct and is kept.

#### DFTB — Slater-Koster tables from `.skf` files, always non-orthogonal

```cpp
struct DftbConfig {
  std::string skf_dir;
  std::vector<std::string> species;   // "C", "H"; reader finds C-C.skf, C-H.skf, ...
  SpinSpec spin;
};
```

This is the piece that does not exist today. The `.skf` format carries the
grid, H and S tables, onsite energies, and the basis size in its header — so
`BasisSpec` is read from the files, not declared. **New work**, listed in §5.

Internally DFTB and SKTB share one evaluator: both resolve to the same
`SkTable` (the current `SKTBModel` struct), so `sktb_construction.hpp`'s
Slater-Koster rotation machinery is reused unchanged. The difference is only
*where the radial values come from* — analytic functions vs tabulated splines —
which is already a variant inside `SKRadialFunction`.

### 2.5 The Hamiltonian file format tag

The explicit ask. Today's h5 layout is one convention among several, hardcoded.

```cpp
// io/hamiltonian_file.hpp
enum class HamiltonianFormat {
  kDeeptbHdf5,     // group "0", datasets "i_j_Rx_Ry_Rz"  <- today's only reader
  // kOpenmxHdf5, kSiestaHdf5, ... added as readers land
};

struct HamiltonianFile {
  std::string path;
  HamiltonianFormat format = HamiltonianFormat::kDeeptbHdf5;
};

// What the file says about itself, so the model can cross-check rather than trust.
struct FileMetadata {
  std::optional<int> n_orbital_total;
  std::optional<SpinSpec> spin;
  std::optional<std::vector<int>> type_norb;
};

class HamiltonianReader {
 public:
  virtual ~HamiltonianReader() = default;
  virtual FileMetadata metadata() const = 0;
  virtual bool ReadBlock(int i, int j, const CellShift& R, DenseBlock* out) const = 0;
};

absl::StatusOr<std::unique_ptr<HamiltonianReader>> OpenHamiltonianFile(
    const HamiltonianFile& file);
```

Adding a format becomes: write one `HamiltonianReader`, register it. Nothing
else changes. `io/hdf5.hpp` becomes `io/hdf5_deeptb.hpp`, one implementation of
the interface.

**Hamiltonian files belong to LCAO only.** `SktbConfig` and `DftbConfig` do not
carry one — computing H from parameters is the entire point of those families.
(See the open question in §6 about the current `sktb.input` / `dftb.input`
override.)

### 2.6 Spin declared once

`SpinSpec` lives on the model, and is the single input to all three consumers:
the `BasisSpec` size multiplier, the file reader's cross-check, and op2c's
`init(ntype, nspin, lspinorb, ...)`. The duplicate copy on `Hdf5Input` is
deleted rather than left to disagree, and `noncolin` — currently dead in both
structs — becomes a real `Treatment::kNonCollinear` enumerator that either works
or is rejected at `Create` time.

### 2.7 Setup becomes four named, testable stages

```cpp
// setup/
AtomicData        BuildAtomicData(const Structure&, const BasisSpec&, MPI_Comm);
OperatorImages    BuildRealSpaceOperators(const Model&, AtomicData&);
KPointOperators   BuildKPointOperators(const OperatorImages&, const KPointSet&,
                                       const PreconditionerOptions&);
SpectralBounds    EstimateSpectralBounds(const KPointOperators&, const BoundsOptions&);
```

Each returns a value that can be constructed and asserted on in a unit test
without running the stage before it. `RSATB` becomes a façade holding the four
results plus the sampling state, and the observables become free functions over
`(KPointOperators, SpectralBounds, Samples)` — testable without MPI setup.

The identical allreduce-then-broadcast block copy-pasted into `compute_dos`,
`compute_ac`, `compute_dc_spectrum`, and `compute_msd_spectrum` becomes one
helper:

```cpp
// Sums per-k results across master_comm, then broadcasts within sub_comm.
void GatherKPointResults(std::vector<double>* values) const;
```

That alone removes ~60 lines of duplicated MPI from `rsatb.hpp`.

### 2.8 Preconditioner: enum, and name the side effect

```cpp
enum class PreconditionerTarget { kNone, kOverlap };

struct PreconditionerOptions {
  PreconditionerTarget target = PreconditionerTarget::kOverlap;
  bool release_overlap_after_build = false;   // was: mode == "H"
  double tol = 1e-7;
  double sptol = 1e-5;
  int maxiter = 20;
  IsqrtmMethod method = IsqrtmMethod::kGraph;
};
```

`"H/S"`, `"H"`, `"S"`, `"None"` and their case sensitivity are gone. The
memory-freeing behaviour is now a field that says what it does.

### 2.9 Python: bind the config structs, delete the marshalling

Replace the `to_dict()` → `parse_*` pipeline with direct `py::class_` bindings
plus `def_readwrite`. One definition of each field instead of four. This deletes
~210 lines of `pybind_rsatb.cpp` and the entire `_split_*` / `_looks_like_*`
layer in `driver.py`, and turns a typo'd key from a silent default into a
`TypeError`. Python keeps only the thin ergonomic builders that genuinely help
(`sk_spline`, `sk_exp_poly`), now returning bound C++ objects.

Target usage — the LCAO case from `test/test_dos.py`:

```python
from rsatb import RSATB, Structure, LcaoModel, HamiltonianFile, OrbitalFiles, \
                  PseudopotentialFiles, RunOptions, PreconditionerOptions

structure = Structure.FromFile("./data/diamond/diamond_sc111.vasp")

model = LcaoModel(
    hamiltonian=HamiltonianFile("./data/diamond/C2.0/hamiltonians.h5", format="deeptb"),
    orbitals=OrbitalFiles(["C_gga_7au_100Ry_2s2p1d.orb"], dir="./data/diamond/"),
    pseudopotentials=PseudopotentialFiles(["C_ONCV_PBE-1.0.upf"], dir="./data/diamond/"),
    spin="none",
)
# model.basis.type_norb     == [13]    derived from the .orb file
# model.basis.cutoff_radius == [3.70]  derived from the .orb file

calc = RSATB(structure, model,
             RunOptions(kpoints=kpoints, nsamples=500,
                        preconditioner=PreconditionerOptions(release_overlap_after_build=True)),
             npar=5)

dos, energy = calc.compute_dos(200)
```

and the SKTB case from the MATBLG transport run (then
`examples/matblg_pz_sktb.py`, now `examples/06_matblg_transport.json`):

```python
model = SktbModel(
    basis={"C": ["pz"]},
    hamiltonian=SkParameters(
        onsite={"C": {"pz": 0.0}},
        pairs={("C", "C"): {"pp_pi": sk_spline(r, vpp_pi, r_cut),
                            "pp_sigma": sk_spline(r, vpp_sigma, r_cut)}},
    ),
)                                          # orthogonal: no overlap= given
calc = RSATB(structure, model, RunOptions(kpoints=[[0, 0, 0]]), npar=args.npar)
```

Five setup entry points on the pybind class (`setup_model`, `setup_lcao`,
`setup_sktb`, `setup_dftb`, `setup_config`) collapse to zero — setup happens in
the constructor. `test_solve` and the empty `band_pardiso` stub
([rsatb.hpp:164](rsatb/rsatb.hpp:164)) leave the public surface.

---

## 3. Google-style conventions to apply

Mechanical, but they are much of what makes the current code hard to scan:

- `CamelCase` for functions, `snake_case_` for members, `kConstantCase` for
  enumerators. Today: `build_operator_set`, `normalized_mode`, `has_overlap_inputs`.
- One class or one closely-related group per header. `sktb.hpp` currently holds
  six concerns in 852 lines; split into `utils/strings.hpp`,
  `io/periodic_table.hpp`, `model/sktb/orbital_labels.hpp`,
  `model/sktb/radial.hpp`, `model/sktb/sk_table.hpp`.
- Prefer enums to strings for closed sets — `PreconditionerTarget`,
  `HamiltonianFormat`, `SpinSpec::Treatment`, `Axis`, `DcFormulation`. The
  `axis_to_index` / `parse_dc_formulation` string parsers go away.
- Return errors, do not infer intent. Every `if (thing.empty()) { silently take
  another path; }` becomes an explicit capability check or a failure.
- Document *why*, not *what*. Most current comments restate the next line; the
  genuinely useful ones (the `derive_atomwise_cutoff_radii` docstring explaining
  the `r_i + r_j` convention, the MSD route-selection comments) are the model.

---

## 4. Migration table

| today | after |
|---|---|
| `RSATB.from_file(npar, f, r_max_per_type, type_norb)` + `setup_sktb(model, opts)` | `RSATB(Structure.FromFile(f), model, opts, npar)` |
| `model.prepare_atomic_spec_from_file(f)` | deleted — derived internally |
| `_prepare_sktb_atomic_spec_from_*` (pybind) | deleted |
| `SKTBModel` / `DFTBModel` (identical aliases) | `SktbModel`, `DftbModel` (genuinely different sources) |
| `Hdf5Input{filename, nspin, lspinorb, noncolin}` | `HamiltonianFile{path, format}` + `SpinSpec` on the model |
| `LCAOIntegralInput{orb_*, psd_*, nspin, ...}` | `OrbitalFiles` + `PseudopotentialFiles` + `SpinSpec` |
| `PreconditionerOptions.mode = "H"` | `.release_overlap_after_build = True` |
| `PreconditionerOptions.mode = "None"` | `.target = kNone` |
| `SetupOptions` | `RunOptions` (unchanged fields) |
| `setup_model` / `setup_lcao` / `setup_sktb` / `setup_dftb` / `setup_config` | constructor |
| `sktb.input` / `dftb.input` h5 override | see §6 |

Observable methods (`compute_dos`, `compute_ac`, `compute_dc*`, `compute_msd*`,
`compute_quantum_metric`) keep their signatures and return shapes.

---

## 5. Work items, in dependency order

1. **`io/`**: split `model/io.hpp` → `io/structure.hpp` + `io/periodic_table.hpp`;
   add `io/hamiltonian_file.hpp` (format tag + reader interface); reshape
   `io/hdf5.hpp` → `io/hdf5_deeptb.hpp` as the first implementation.
2. **`model/basis_spec.hpp`**: `BasisSpec`, `SpinSpec`, `Capabilities`.
3. **`model/model.hpp`**: the abstract interface.
4. **`LcaoModel`**: derive `BasisSpec` from op2c; make orbitals required;
   cross-check against `FileMetadata`. *Deletes the hand-transcribed constants.*
5. **`SktbModel`**: split `sktb.hpp` by concern; one input dialect; drop the
   `_split_*` shape-guessing.
6. **`setup/`**: the four stages + `GatherKPointResults`.
7. **`RSATB`**: single constructor, façade only.
8. **pybind + `driver.py`**: bind the structs, delete the marshalling.
9. **`io/skf.hpp` + `DftbModel`**: the genuinely new capability. Independent of
   1–8 and can land last.

Steps 1–8 are restructuring with no physics change, so the existing tests
(`test/test_dos.py`, `test/test_graphene_pz_dos.py`, and the MATBLG transport
run) are the regression suite — same numbers before and after. Step 9 needs new reference data.

### Status

All planned steps are implemented.

Landed in step 7b:

- `setup/kpoint_partition.hpp` — `PartitionKPoints` is pure arithmetic on ranks
  and counts, testable with no operators built at all.
- `setup/spectral_bounds.hpp` — `EstimateSpectralBounds` is a free function over
  a `BoundsKPointView` per k-point. It takes a *view* rather than the owning
  state, so it does not need to know how the operators were built or who keeps
  them alive; that is what let it move without relocating the lazy velocity and
  position cache, which was the risky part of the original plan and turned out
  to be unnecessary.
- `OperatorSet::initialize` now reads as partition → sample → precondition →
  `resolve_spectral_bounds`, with the last two expressible independently. The
  most expensive part of setup (two PRIMME solves per k-point) is no longer
  entangled with k-point distribution.
- `test_setup_stages` (8 tests) sweeps rank layouts to assert every k-point is
  owned exactly once and that ranks sharing a sub-communicator agree, plus the
  bounds validation and failure paths.
- `calc/eigen.hpp` had `using namespace rsatb::backend;` without that namespace
  being declared, so it only compiled when some earlier include happened to open
  it. Including it from a new standalone test surfaced this; now declared.

Landed in step 7c:

- `calc/observables.hpp` holds the six `compute_*` bodies as free functions over
  an `ObservableContext`, together with `SampleVectors` (was `RSATB::SamplingState`),
  the Chebyshev factory, the axis helpers, and the two gather helpers.
- `rsatb.hpp` drops from 934 to 408 lines and is now what it claims to be: MPI
  layout, construction, and six one-line forwarders. The public signatures are
  unchanged, so pybind and every test needed no edit.
- The observables no longer need a constructed `RSATB` to run: an
  `ObservableContext` is an operator set, an atomic data, sample vectors, and
  two communicators.

Landed in step 7a:

- `GatherKPointResults` and `BroadcastSharedGrid` replace five copies of the
  allreduce-then-broadcast tail in `compute_dos`, `compute_ac`,
  `compute_dc_spectrum`, and `compute_msd_spectrum`.
- `compute_msd_spectrum` used to SUM-reduce its energy grid across leaders,
  which was correct only because non-Gamma ranks left it zero-filled.
  `BroadcastSharedGrid` broadcasts from the rank that computed it, which also
  fixes a latent bug in `compute_dc_spectrum`: a sub-communicator owning no
  k-point previously returned an all-zero energy axis.
- `RSATB::AtomicGraphArrays` now validates that the local atom and edge counts
  sum to the declared totals across the sub-communicator. Passing the full
  graph to every rank — the natural misreading of a distributed description —
  used to build the system once per rank and silently double the spectrum;
  it now raises.

Landed in step 9:

- `io/skf.hpp`: parses DFTB `.skf` files — grid header, homonuclear onsite
  block, and `nGridPoints` rows of 20 values. `SkfChannelLayout()` is the one
  place mapping the file's ten columns onto `(row_l, col_l, bond_type)`.
  Hartree→eV and Bohr→Å conversion happens once, in the reader; overlap
  integrals are dimensionless and deliberately left unscaled.
- `model/sktb/skf_table.hpp` turns a directory of `.skf` files into
  `SkParameters`, and `DftbModel::FromSkfDirectory` builds a model from it.
  Because step 5 separated the input dialect from the resolved `SkTable`, the
  rotation machinery, image construction, and every observable are reused
  unchanged — the reader only has to produce `SkParameters`.
- The basis is declared, not inferred. A `.skf` tabulates all ten channels up
  to *d* regardless of which shells the parameterization uses, and DFTB+ also
  takes the basis from its own input. Inferring it from "is `E_d` numerically
  zero" would be a guess.
- `test_skf` (5 tests) covers the grid and unit conversion, the column-to-channel
  mapping, and a full round-trip: a synthetic `.skf` produces the same
  Hamiltonian, onsite and off-diagonal, as an equivalent hand-built model.

**Heteronuclear pairs: resolved without external data.** A `.skf` column such as
`Hsp0` is one number for the unordered pair {s, p}, and for an ordered species
pair it could mean either assignment. The resolution came from
`copy_shell_block_to_pair_block`, which matches orbitals by shell: for C(s,p)
with H(s) a p on hydrogen does not exist, so the column can *only* be the p on
carbon with the s on hydrogen. **The reading is forced by the declared basis, not
assumed.** The ambiguity is real only when both species carry both shells of the
pair — C(s,p) with N(s,p) — and only that case is refused.

`TwoSpeciesCarbonHydrogenBuildsThePhysicalSigmaBond` is the decisive check: in a
C–H dimer along z, the p_z–s coupling must be non-zero while p_x and p_y stay
zero, and the reverse block must be the conjugate transpose. Had the column been
read onto the hydrogen side it would have been dropped as out-of-basis and that
coupling would vanish — so the test fails loudly on the wrong reading. No DFTB+
reference was needed.

Landed in steps 1–4:

- `io/periodic_table.hpp`, `io/structure.hpp` — `model/io.hpp` split by concern;
  the duplicate symbol table in `sktb.hpp` now delegates here.
- `io/hamiltonian_file.hpp`, `io/hdf5_deeptb.hpp`, `io/hamiltonian_readers.hpp` —
  format tag, reader interface, registry, and the DeePTB implementation.
  `OperatorConstruction::build_from_hdf5` became `build_from_reader`.
- `model/basis_spec.hpp`, `model/model.hpp`, `model/operators.hpp` — `BasisSpec`,
  `SpinSpec`, `Capabilities`, and the `Model` interface.
- `model/lcao_model.hpp` — derives `type_norb` and `cutoff_radius` from the
  orbital files; orbitals are required, so LCAO is non-orthogonal by
  construction.
- `RSATB(Structure, Model, SetupOptions, npar)` and
  `build_operator_set(..., const Model&, ...)`.
- `PreconditionerTarget` enum + `release_overlap_after_build`.
- `sktb.input` / `dftb.input` removed; supplying it is now an explicit error.
- pybind: config structs bound with `py::class_`; `parse_hdf5_input`,
  `parse_lcao_integrals`, `parse_lcao_model`, and `parse_setup_options` deleted
  (~140 lines), along with `setup_lcao` and `setup_config`.

Landed in step 5:

- `model/sktb/` split by concern: `orbital_labels.hpp` (label and channel
  vocabulary), `radial.hpp` (`SkRadialFunction` plus `MakeSpline` /
  `MakeExpPoly`), `sk_table.hpp` (input `SkParameters` and resolved `SkTable`).
  `sktb.hpp`'s six concerns are gone, and the SK types moved from
  `rsatb::backend` to `rsatb::model::sktb`, so the `model` ↔ `backend`
  namespace interleaving in §1.8 is resolved.
- `SktbModel` and `DftbModel` implement `Model`. **One input dialect**: H and S
  are separate `SkParameters`, so there is nothing to disambiguate. The species
  order is derived from the declared basis rather than passed separately.
- `Model::ProvidesOperatorBuilders()` states where velocity and position come
  from. `install_default_fallbacks` — which silently changed available physics
  based on whether an overlap happened to be null — is deleted.
- `AnyModelConfig`, `SKTBModelConfig`, `DFTBModelConfig`, `build_model_artifacts`,
  `validate_pairwise_model_setup` (the duplicated cross-check from §1.7), and
  the `setup_model` / `setup_sktb` / `setup_dftb` pybind methods are gone. Setup
  happens in the constructor for every family.
- `RSATB::AtomicGraphArrays` replaces the raw-array constructor. `type_norb` is
  no longer an argument — it comes from the model's basis.
- `TightBindingFileModel`: the fourth combination flagged in §6 — an orthogonal
  Hamiltonian read from a file. It turned out to be load-bearing in
  `test_operator_api.cpp`, where it had been expressed as an LCAO model with no
  orbital files, i.e. the silently-orthogonal bug from §1.3. Giving it its own
  model keeps each family's story intact. It is the one model that must be told
  its basis, because the supported formats carry no basis header.
- `driver.py`: 616 → 347 lines. `_matrix_key`, `_split_onsite`, `_split_pairs`,
  `_looks_like_matrix_group`, `_lower_pair_channels`, `_pair_species`,
  `_radial_config`, `_build_sktb_native_config`, and both `to_dict()` layers
  deleted. `pybind_rsatb.cpp` lost a further 154 lines of `parse_sktb_*`.

Verification:

- `test/test_dos.py` runs on 5 ranks and reports `type_norb=[13]`,
  `cutoff_radius=[3.7042]`, derived — matching the previously hand-supplied
  `[13]` and `[3.7]`. The SKTB basis derivation likewise reproduces the old
  `prepare_atomic_spec_*` output exactly (`type_norb=[1]`, `cutoff=[0.8]`).
- Spectral bounds are **bit-identical** before and after, for both families:
  LCAO `emin=-8.621686512892`, `emax=200.345072902732` (exercising the h5 read,
  the op2c overlap, and the preconditioner) and SKTB `emin=-8.6`,
  `emax=8.0625` (exercising the SK table build and Hamiltonian images).
- The DOS values themselves are not comparable run to run: the random sampling
  is unseeded, and the old-vs-new spread (1.67) sits inside the same-code
  run-to-run spread (old-vs-old 1.58).
- Full C++ suite green, including the 21 SK rotation-math tests in `test_sktb`
  and the 14 operator-API tests. `test_binding.py`,
  `test_graphene_pz_dos.py`, and `test_dos.py` pass.

Two C++ tests changed meaning rather than form, and are noted here because the
coverage moved:

- `SetupRejectsSKTBWhenStoredCutoffIsTooSmall` became
  `SktbGraphCutoffIsDerivedFromThePairCutoff`. It asserted that a
  caller-supplied graph cutoff disagreeing with the model's pair cutoffs is
  rejected; the cutoff now comes from the model, so the two cannot disagree.
  What is worth asserting is the derivation, which is what it now checks.
- `ComputeMSDSpectrumFailsClearlyForPeriodicOverlapSystems` became
  `ComputeMSDSpectrumFailsClearlyForOverlapSystems`, using an overlap-carrying
  SKTB model. The old form combined a file Hamiltonian with op2c orbital files
  under a single config, which no longer type-checks.

### 7.3 The neighbour list dropped bonds on a rounded cell boundary

Found while checking the JSON layer's `structure.supercell` against an analytic
answer. Graphene tiled 2×2 reported |E|max = 7.60 eV and 4×4 reported 7.81,
where every supercell must give the same 3|t| = 8.10 as the primitive cell.
Nothing was raised: the run completed and returned a narrower spectrum.

`NeighborList::build` wraps fractional coordinates into the cell for binning.
`s - std::floor(s)` is not guaranteed to land in `[0, 1)`: for an `s` that is a
tiny **negative** number — `-3.4e-17`, which is what an exactly-zero coordinate
computes to through `dot(r, bxc)/det` — the subtraction rounds up to exactly
`1.0`. The bin index was then taken with `% nbins`, which sent that atom to the
bin at the **near** face while its wrapped position sat at the **far** face.
The search scans a fixed neighbourhood of bins around each atom, so it looked a
whole lattice vector away from where the atom actually was, and found nothing.

Two changes, one rule: **an atom's bin must contain its working position.**
`wrap` folds once more when the result reaches 1.0, and the bin index is
*clamped* to the nearest bin rather than wrapped modulo — a coordinate at the
top of the cell belongs in the last bin, not the first.

The trigger is a rounding outcome, so it was build- and geometry-dependent: the
same 3×3 graphene cell was correct when its coordinates were accumulated one
way and wrong when accumulated another, which is what made it look like two
separate defects. `RSATB.from_graph` was never affected — a caller-supplied
neighbour list bypasses the search entirely, which is what the bisection used
as the reference.

Verification: `test_neighbourlist` gains three tests, all of which fail on the
old code — a graphene supercell sweep asserting three neighbours per atom, a
deterministic minimal case (an atom at fractional `-1e-18`, whose bond across
the boundary used to vanish), and a randomized comparison against brute force
over 1000 skewed cells with atoms on faces and whole cells outside. Before the
fix that sweep loses bonds in 5% of cases. At the Python level,
`test/test_supercell_bonds.py` pins the physics: 3|t| for supercells 1×1
through 5×5, and the same answer whether the basis atom is written inside the
cell or at its equivalent site outside. The LCAO diamond reference is
bit-identical either way (`emin = -8.621686512892`), so that path never tripped
the boundary case.

### Bug found and fixed along the way

`OperatorConstruction::orbital_cutoffs` and `projector_cutoffs` returned op2c's
radii in **Bohr** and passed them straight to `AtomicData::get_graph3b`, which
works in **Angstrom**. The three-body velocity graph was therefore built with
cutoffs inflated by ~1.89×. Results were unaffected — a larger cutoff yields a
strict superset of the required triples — but the graph was substantially larger
than necessary. Both functions now convert.

Two smaller fixes in the same area: `build_from_reader` no longer shares a
`found` flag across OpenMP threads (a data race in the old onsite loop), and it
now checks onsite block dimensions, which the old code skipped because it passed
`inorb` as both input and output to the reader.

---

## 6. Resolved: the h5 override

`SktbModelConfig` and `DftbModelConfig` each carried an
`std::optional<Hdf5Input> input` that overrode the parameterized Hamiltonian
with one loaded from h5. Under this design, "H comes from a file" is the LCAO
story and "H comes from parameters" is the SKTB/DFTB story, so the override had
no natural home.

The override itself is dropped — supplying it is now an explicit error. But the
combination it enabled, an h5 Hamiltonian with **no** overlap, turned out to be
real: it is what `test_operator_api.cpp` used throughout, expressed as an LCAO
model with no orbital files, which is exactly the silently-orthogonal bug from
§1.3. It is now `TightBindingFileModel` — a fourth family with a coherent story
of its own (orthogonal, file-backed, no parameters), which is also what a DeePTB
Slater-Koster export produces.

---

## 7. Resolved: two defects found while writing the examples

Neither was introduced by this redesign; both were found by checking the new
examples against analytic answers rather than merely confirming they ran.

### 7.1 Neighbour shifts disagreed with the positions they indexed

`NeighborList::build` wraps positions into the cell so it can bin them, and
returned shift vectors relative to that internal wrapped copy. Both callers —
`process_input_rank0` and `from_distributed` — then re-tested the bond length
using the *unwrapped* input positions with those shifts. For an atom whose
fractional coordinate lay outside [0, 1) the two disagree by a lattice vector,
so the bond came out at the wrong length and was silently dropped by the cutoff
test. Graphene written with its conventional basis — the second carbon at
fractional (-1/3, 2/3) — lost two of its three bonds, and |E|max came out 2.700
instead of 8.100 with no warning.

The fix keeps the caller's coordinates untouched and corrects the shift instead:
wrapping records the lattice translation `W` it applied to each atom, and a
neighbour is reported as `R + W_j - W_i`, which is the shift measured against
the input positions. For a structure already inside the cell every `W` is zero
and the result is bit-identical to before.

Wrapping the positions instead would have been the smaller edit, and was
rejected: atom order and coordinates are what a Hamiltonian file's own atom
indexing and R-vector convention refer to, so moving an atom would silently
re-aim every file lookup. **The convention is now stated once and holds end to
end — input coordinates and atom order reach the vbcsr graph unmodified, and
every shift is expressed against them.** `RSATB(AtomicGraphArrays, ...)` already
required the caller's positions and shifts to be mutually consistent; the
derived path now means the same thing by the same rule.

The cost of correctness here is that `R` can grow: an atom `n` cells outside
the box produces shifts of order `n`. The *number* of images does not blow up —
it is bounded by the bond count, which is geometry — so this changes the labels,
not the memory. Measured on graphene with one atom displaced by ten cells:
6 edges and 6 distinct images either way, `max|R|` rising from 1 to 11.

### 7.2 Setup hung when a cell had fewer atoms than ranks

Two atoms on two ranks hung. The graph partitioner was the obvious suspect and
the wrong one: `AtomicData::from_points` completes, and so do the Chebyshev
matvecs. Narrowing by rank layout turned up **two independent defects**, in
different regimes.

**A rank owning exactly one scalar row deadlocks PRIMME.** The hang is in the
solve inside the spectral-bounds estimate, and the boundary is sharp: two rows
per rank is fine, one is not. Confirmed on 1D chains across layouts -- nLocal=1
hangs at every size tried, nLocal>=2 always completes.

A matrix that thin is also tiny: one row per rank means the global dimension is
below twice the rank count. `setup/dense_bounds.hpp` detects the condition
collectively and diagonalizes such a matrix outright -- materialized as `op * I`,
gathered to rank 0, `zheev` or `zhegv` -- which is both cheaper and more
accurate than the iterative solve it replaces. Above
`kMaxDenseBoundsDimension` it raises rather than allocating unboundedly. The
LAPACK integer width comes from vbcsr's `vbcsr_lapack_int`, not `int`, because
an ILP64 BLAS is auto-detected at configure time and a mismatch would corrupt
silently.

**A rank owning no atoms at all deadlocks the operator stack.** Reachable only
by giving a sub-communicator more ranks than the cell has atoms. Two smaller
problems sit underneath it: ParMETIS was handed a null `part` pointer on empty
ranks (fixed -- it gets a dummy element, like `adjncy` already did), and the
non-orthogonal operator build hangs outright. Rather than audit every collective
in that stack for a layout nobody wants, `RSATB::RequireEveryRankOwnsAtoms`
rejects it with a message naming the fix: use fewer ranks, or raise `npar` so
the extra ranks work on k-points, where they are useful.

The rule that now holds: **setup either works or says why. It never hangs.**
Two atoms on one or two ranks return the same spectral window; three or more
ranks fail immediately with an explanation.

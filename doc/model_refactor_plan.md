# Model-Centric Refactor Plan

## Summary

This refactor will replace the current source-centric operator setup with a
model-centric setup that matches the project code style: concrete, flat, and
easy to extend.

The core direction is:

1. `AtomicData` is built first by IO or future atomic-building tools.
2. A single `model` is selected to define operator semantics.
3. The model constructs or loads `H` and optional `S`, then supplies
   model-specific operator behavior and fallback rules.

The public model families are:

- `lcao`
- `sktb`
- `dftb`

`HDF5` is not treated as a model family. It is an input mode for any model
family, used to load Hamiltonian and overlap matrices for the selected model.

`dftb` is not a separate matrix-source path. It is a native model family whose
native input is a spline-based pairwise-TB parameterization using the same
SKTB-like angular transform rules to construct both Hamiltonian and overlap.

This refactor is intentionally a hard break. Old setup entry points and old
`sources.*` config shapes will be removed instead of preserved through
adapters.

---

## Design Goals

- Make the architecture flat and model-first.
- Separate `AtomicData` construction from model/operator construction.
- Remove hidden mutable state from user-facing config.
- Keep public model names semantically meaningful for future development.
- Allow `sktb` and `dftb` to share pairwise-TB internals without merging them
  in the API.
- Keep HDF5 loading available for all model families.
- Avoid speculative abstraction layers such as provider stacks unless a
  concrete need appears later.
- Reduce the number of public setup concepts rather than replacing one complex
  setup path with another.

---

## Simplification Rules

These rules are mandatory for the implementation. If a change violates them, it
should be treated as a design regression even if it compiles.

- Do not replace the current provider stack with another general framework.
- Do not introduce a new public catch-all wrapper such as a public
  `ModelSetupConfig` if `SetupOptions` plus typed model configs already express
  the same information.
- Do not keep parallel validation logic for typed setup and dict-based setup.
  The structured config path must only parse and forward.
- Do not create a standalone `hdf5` model or standalone HDF5 provider branch.
  `Hdf5Input` is only an input helper reused by model families.
- Do not split SKTB and DFTB into separate low-level assembly engines if the
  only difference is radial parameterization. Share the pairwise-TB core.
- Do not add model-specific convenience APIs that bypass the common typed setup
  path. Named model methods may exist, but they must forward into the same core
  implementation.
- Do not keep mutable resolved state inside public config structs.
- The net result should remove code and concepts:
  - delete `IOperatorProvider`
  - delete `OperatorSetBuilder`
  - delete provider factories
  - delete source-centric setup structs
  - delete legacy dict-only setup paths

- Public setup concepts after the refactor should be limited to:
  - `SetupOptions`
  - `PreconditionerOptions`
  - `LCAOModelConfig`
  - `SKTBModelConfig`
  - `DFTBModelConfig`
  - shared helper inputs such as `Hdf5Input`

---

## Public API Shape

The setup API should be function-call-first.

Structured dict / JSON-like config remains supported because it is useful for
scripts, config files, and interoperability, but it must be a thin layer built
on top of the normal C++ and Python interfaces instead of replacing them.

The new setup design has exactly one active model per call.

### API layers

1. Primary typed API
   - C++ uses normal setup functions with concrete config structs.
   - Python uses normal methods and small helper classes mirroring the C++
     structs.
2. Structured config API
   - Python dict / JSON-like config is accepted as a convenience layer.
   - It is parsed into the same typed setup objects, then forwarded to the same
     typed setup path.
3. Internal model resolution
   - All public entry points end in the same model-resolution and
     operator-construction path.

### Shared setup types

Common run options should be separated from model definition.

Suggested shared types:

```cpp
namespace rsatb::model {

struct PreconditionerOptions {
    std::string mode = "H/S";
    double tol = 1e-7;
    double sptol = 1e-5;
    int maxiter = 20;
    std::string method = "graph";
};

struct SetupOptions {
    std::vector<std::vector<double>> kpoints = {{0.0, 0.0, 0.0}};
    PreconditionerOptions preconditioner;
    int nsamples = 1;
    bool compute_bounds = true;
};

} // namespace rsatb::model
```

These types should exist in both C++ and Python with closely matching names and
fields.

There should be no additional public top-level wrapper config around them.

### C++ primary API

The main C++ interface should use normal functions with typed model configs.

```cpp
namespace rsatb::model {

struct LCAOModelConfig;
struct SKTBModelConfig;
struct DFTBModelConfig;
using AnyModelConfig = std::variant<LCAOModelConfig, SKTBModelConfig, DFTBModelConfig>;

} // namespace rsatb::model

class RSATB {
public:
    void setup(const model::LCAOModelConfig& model, const model::SetupOptions& options = {});
    void setup(const model::SKTBModelConfig& model, const model::SetupOptions& options = {});
    void setup(const model::DFTBModelConfig& model, const model::SetupOptions& options = {});
    void setup(const model::AnyModelConfig& model, const model::SetupOptions& options = {});
};
```

Design rules:

- The typed overloads are the primary stable API.
- The `AnyModelConfig` overload exists as a generic convenience entry point, not
  as the only public path.
- Users should be able to construct model configs directly in normal C++ code
  without going through serialized config.

### Python primary API

Python should mirror the C++ typed interface with normal methods and small
config/helper classes.

```python
opts = rsatb.SetupOptions(
    kpoints=[[0.0, 0.0, 0.0]],
    nsamples=1,
    compute_bounds=True,
)

model = rsatb.SKTBModel(
    input=rsatb.SKTBNativeInput(...),
)

rs.setup_sktb(model, opts)
# or
rs.setup_model(model, opts)
```

The Python surface should provide:

- `rsatb.SetupOptions`
- `rsatb.PreconditionerOptions`
- `rsatb.LCAOModel`
- `rsatb.SKTBModel`
- `rsatb.DFTBModel`
- input/helper classes such as `Hdf5Input`, `LCAOIntegralInput`,
  `SKTBNativeInput`, and `DFTBNativeInput`
- setup methods:
  - `setup_lcao(...)`
  - `setup_sktb(...)`
  - `setup_dftb(...)`
  - `setup_model(...)`

Design rules:

- Python should not force raw dict construction for normal use.
- Python helper classes should map closely to C++ structs so documentation and
  behavior stay aligned.
- `setup_model(...)` is generic convenience; named model methods remain useful
  and should stay available.

### Structured config API

The structured config remains supported, but only as a forwarding layer.

Suggested Python shape:

```python
{
  "kpoints": [[...], ...],
  "nsamples": 1,
  "compute_bounds": True,
  "preconditioner": {
    "mode": "H/S",
    "tol": 1e-7,
    "sptol": 1e-5,
    "maxiter": 20,
    "method": "graph",
  },
  "model": {
    "kind": "lcao" | "sktb" | "dftb",
    "input": {
      "kind": "native" | "hdf5",
      ...
    },
    ...
  }
}
```

Behavior rules:

- `setup_config(...)` or `setup_dict(...)` is convenience only.
- It must build the same typed `SetupOptions` and typed model config objects
  that the normal APIs use.
- Validation, defaults, and error messages must come from the shared typed path,
  not from a separate dict-only implementation.
- The structured config is useful for serialization and workflow exchange, but
  it is not the canonical API shape for library users.
- It must not introduce any extra public config class beyond the typed setup
  types already listed above.

### Model family meaning

- `lcao`
  - Model based on two-center integrals.
  - Overlap, velocity, and position semantics come from the LCAO formulation.
  - Supports `input.kind = "hdf5"` for loading `H` and optional `S`.
  - Keeps the existing two-center integral data for operator construction.

- `sktb`
  - Model based on explicit tight-binding construction from onsite and hopping
    formulas/tables.
  - Uses simpler operator rules and existing orthogonal fallbacks where
    applicable.
  - Supports `input.kind = "native"` and `input.kind = "hdf5"`.

- `dftb`
  - Separate public model family.
  - Native DFTB is a spline-based pairwise-TB model.
  - It uses the same SKTB-like angular transform parameterization to construct
    both Hamiltonian and overlap from radial spline data.
  - Supports `input.kind = "native"` and `input.kind = "hdf5"`.

### Hard break scope

The following public paths are removed in this refactor:

- old `setup(Hfile, orb_names, orb_dir, psd_names, psd_dir, ...)`
- old dict-only `setup_sktb(kpoints, sktb_config, ...)`
- old `setup_config(...)` schema based on `sources.*`
- `from_file_sktb(...)`
- `from_points_sktb(...)`
- old `backend/...` compatibility naming

The following public paths replace them:

- typed C++ setup overloads by model config type
- typed Python setup methods by model family
- one generic structured-config entry point using the new `model` schema

---

## Internal Architecture

### 1. Replace source-centric setup with model-centric setup

Current setup is organized around sources like:

- `hdf5`
- `op2c`
- `sktb`

This will be replaced with one model-centric resolution path:

1. `resolve_model(AtomicData, AnyModelConfig) -> ResolvedModel`
2. `build_model_artifacts(AtomicData, ResolvedModel) -> ModelArtifacts`
3. `finalize_operator_set(ModelArtifacts, SetupOptions, runtime) -> OperatorSet`

This keeps the flow flat and explicit.

This path must be the only internal setup path. Do not add a second internal
"structured setup" or "source setup" pipeline beside it.

### 2. Remove provider stack

The current `IOperatorProvider` / `OperatorSetBuilder` / provider factory design
will be removed.

These abstractions are heavier than needed for the actual problem and make the
setup harder to understand. The replacement is one direct build path driven by
the selected model family.

### 3. Keep `OperatorSet` as the runtime cache

`OperatorSet` remains the runtime object used by later calculations, but it
should no longer own the model-selection logic.

Its responsibility becomes:

- store sampled `H(k)` and optional `S(k)`
- lazily construct derived operators where needed
- cache runtime objects per k-point
- expose capability flags

### 4. Introduce `ModelArtifacts`

A flat internal struct will hold the result of model construction:

- Hamiltonian images
- optional overlap images
- optional velocity builder
- optional position builder
- capability flags
- any model metadata needed at runtime

This replaces cross-provider registration.

`ModelArtifacts` is an internal assembly result, not a public setup concept.

### 5. Make fallback policy explicit and centralized

Fallback logic should live in one place, after model artifacts are built.

Examples:

- orthogonal velocity fallback from Hamiltonian images
- orthogonal diagonal position fallback
- first-lattice-moment support

These are runtime/operator policies, not separate providers.

---

## Model-Specific Internal Design

### LCAO model

The LCAO model owns:

- overlap construction from two-center integrals
- velocity construction from the existing LCAO formulation
- position construction from the existing two-center machinery
- any non-orthogonal operator rules already documented in the repo

For `input.kind = "hdf5"`:

- HDF5 provides `H`
- if overlap is present in the file, it provides `S`
- LCAO operator semantics still apply
- two-center inputs remain the source of explicit position/velocity operators

This keeps HDF5 as data input, not as model semantics.

### SKTB model

The SKTB model owns:

- basis expansion
- onsite and pair-channel resolution
- explicit TB block construction from formulas/tables
- simple operator behavior
- orthogonal fallback behavior when no model-specific operator exists

Important cleanup:

- remove `resolved_type_symbols` from public config
- move all resolved/internal state into dedicated resolved structs
- keep user config immutable during setup

### DFTB model

The DFTB model becomes a separate public family now.

It should not remain mixed into SKTB semantically, but it also should not be
implemented as a completely separate low-level engine.

DFTB owns:

- its own public config type
- its own resolver entry point
- its own dispatch branch in model setup
- its own model identity and future operator semantics

Native DFTB is defined as:

- a spline-based pairwise-TB model
- using SKTB-like angular transform rules
- building both Hamiltonian and overlap from native radial spline data
- differing from SKTB mainly in radial parameterization and native input shape

In the first refactor:

- `input.kind = "hdf5"` must work
- native DFTB config/resolution and image construction must exist
- native DFTB should reuse the shared pairwise-TB assembly path rather than
  duplicating SKTB logic

### Typed model config design

Each public model family should have its own typed config object instead of
sharing one dict-shaped catch-all interface.

Suggested direction:

- `LCAOModelConfig`
  - contains an `input` field for Hamiltonian source
  - contains the LCAO-specific integral/operator inputs
- `SKTBModelConfig`
  - contains either `SKTBNativeInput` or `Hdf5Input`
- `DFTBModelConfig`
  - contains either `DFTBNativeInput` or `Hdf5Input`

Shared input helpers:

- `Hdf5Input`
- `LCAOIntegralInput`
- `SKTBNativeInput`
- `DFTBNativeInput`

This keeps each model family semantically separate while still allowing reuse of
common input/helper types.

`DFTBNativeInput` should explicitly encode spline/radial inputs for both
Hamiltonian and overlap channels under the DFTB model family. It should not be
modeled as "SKTB with a flag."

### Shared pairwise-TB core

A small shared internal layer should be extracted adjacent to SKTB/DFTB.

This shared layer can contain:

- basis/orbital expansion helpers
- SKTB angular transform helpers
- radial table evaluation
- radial spline evaluation
- pair-parameter lookup
- pair-block evaluation
- image assembly from explicit pairwise TB data

This is an internal reuse layer only. It must not replace the separate public
`sktb` and `dftb` model identities.

---

## AtomicData Interaction

`AtomicData` construction is separate by default.

The intended workflow is:

1. user builds `AtomicData` through structure IO or future atomic-building tools
2. selected model validates against the existing graph, basis counts, and type
   ordering
3. model constructs operators on top of that `AtomicData`

Model-derived atomic preparation remains optional helper functionality outside
the main flow.

That means:

- core `RSATB` should not require model config to construct `AtomicData`
- SKTB-style helper routines for deriving `type_norb` or cutoffs can remain as
  utility helpers if still useful
- those helpers should not define the main architecture

---

## HDF5 Semantics

HDF5 is an input mode for any model family.

### HDF5 rules

- HDF5 supplies Hamiltonian `H`
- HDF5 may supply overlap `S`
- HDF5 does not define model family
- HDF5 does not automatically define velocity or position semantics
- model semantics still decide how velocity/position are built or whether
  orthogonal fallbacks are allowed
- HDF5 must be represented by one shared `Hdf5Input` helper reused by all model
  configs
- HDF5 must not have its own provider, dispatcher family, or standalone setup
  branch

### Behavior when overlap is missing

- if HDF5 does not provide `S`, setup is treated as orthogonal unless the
  selected model explicitly constructs `S` from native model data
- fallback rules depend on the selected model family

---

## File And Naming Refactor

### Namespace and naming

- move from `rsatb::backend` to `rsatb::model`
- remove `backend/` compatibility wrapper headers
- update includes and tests to the new naming directly

### Suggested internal organization

Keep the structure flat and near-neighbor:

- `model/setup/...` for model config + resolution
- `model/lcao/...`
- `model/sktb/...`
- `model/dftb/...`
- `model/shared_pairtb/...` only if the shared code becomes large enough to
  justify one adjacent module

Do not reintroduce a deep layered framework.

---

## Migration Plan

The refactor should be done in these stages:

### Stage 1: introduce new model schema and core internal path

- add typed public setup structs:
  - `SetupOptions`
  - `PreconditionerOptions`
  - `LCAOModelConfig`
  - `SKTBModelConfig`
  - `DFTBModelConfig`
- add `ResolvedModel`
- add `ModelArtifacts`
- add typed C++ setup overloads and Python setup methods
- add direct model dispatch by `model.kind`
- keep behavior equivalent where possible
- do not add a public `ModelSetupConfig`

### Stage 2: add structured-config forwarding on top of typed setup

- implement `setup_config(...)` or `setup_dict(...)` using the new `model` schema
- parse structured config into the same typed setup objects used by the normal
  APIs
- keep all validation/defaulting in the shared typed path

### Stage 3: remove old source-centric internals

- remove provider interfaces
- remove provider factories
- remove source-centric setup structs
- move fallback policy to one central finalize step

### Stage 4: split SKTB and DFTB semantically

- keep SKTB public config focused on SKTB
- create DFTB public config slot and internal resolver path
- implement native DFTB spline-based Hamiltonian and overlap construction
- extract one shared pairwise-TB core for SKTB/DFTB angular/radial assembly

### Stage 5: clean public surface

- remove old legacy setup entry points
- remove old Python wrappers and dict-only legacy signatures
- remove old `sources.*` config parsing
- update examples and docs to prefer typed function-call APIs, with structured
  config shown as a secondary convenience path

---

## Test Plan

### Typed API and dispatch

- C++ typed setup works for `LCAOModelConfig`
- C++ typed setup works for `SKTBModelConfig`
- C++ typed setup works for `DFTBModelConfig`
- Python typed setup works for `LCAOModel`
- Python typed setup works for `SKTBModel`
- Python typed setup works for `DFTBModel`
- generic `setup_model(...)` dispatch matches the named model entry points

### Structured config forwarding

- parse valid `model.kind = "lcao"` configs
- parse valid `model.kind = "sktb"` configs
- parse valid `model.kind = "dftb"` configs
- reject missing or multiple model selections
- reject old `sources.*` config shape with clear error messages
- confirm structured config and typed setup produce the same behavior
- confirm structured config forwards into the same validation and error path

### LCAO behavior

- preserve current HDF5 + two-center-integral path
- preserve overlap construction
- preserve explicit velocity construction
- preserve explicit position construction
- preserve transport behavior for current LCAO-supported cases

### SKTB behavior

- preserve orthogonal SKTB transport behavior
- preserve orthogonal velocity fallback from Hamiltonian images
- preserve orthogonal diagonal position fallback
- preserve current cutoff / type-count validation against `AtomicData`
- confirm no public config mutation during setup

### DFTB behavior

- `dftb + hdf5` dispatch works
- native DFTB builds both Hamiltonian and overlap from spline-based native input
- native DFTB uses the same angular transform conventions as SKTB
- native DFTB and SKTB share the common pairwise-TB assembly path without
  diverging user-facing semantics

### Runtime/operator behavior

- capability flags remain correct
- `H(k)` and optional `S(k)` sampling still work
- lazy caching of velocity, position, and first-lattice-moment still works
- orthogonal fallback behavior remains centralized and testable

### API break coverage

- removed legacy entry points fail clearly
- removed wrapper headers/namespaces are no longer referenced in tests/examples
- documentation examples prefer typed APIs and keep structured config as a
  secondary path

---

## Assumptions

- exactly one model family is active per setup
- `AtomicData` is built first in the main workflow
- HDF5 is an `H/S` input mode, not a model family
- `sktb` and `dftb` remain distinct public APIs
- hard break is acceptable for this refactor
- typed C++ and Python setup interfaces are the primary user-facing APIs
- native DFTB construction is part of the refactor scope

---

## Open Questions For Iteration

These are the main things to refine before implementation:

- exact field names for DFTB spline channels and table containers
- exact Python helper-class ergonomics for typed setup
- whether `lcao` should have a future native tight-binding input mode or stay
  two-center-integral-centered
- final folder names for shared SKTB/DFTB internals
- whether HDF5 overlap should be mandatory, optional, or model-dependent in
  each family

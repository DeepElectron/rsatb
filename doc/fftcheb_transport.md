# FFTCheb Bulk Transport in RSATB

This note explains how the current FFTCheb bulk-transport implementation is wired into `rsatb`, what conventions it uses, and how the paper maps onto the code.

The implementation targets the bulk transport part of:

- `FFTCheb.pdf`
- `FFTChebsuppInfo.pdf`

Scope of the current implementation:

- full-spectral bulk DC transport
- Kubo-Greenwood longitudinal response
- Kubo-Bastin conductivity tensor, including Hall-type terms
- non-orthogonal basis support from the start

Out of scope in the current code:

- two-terminal conductance
- lead/contact/absorbing-boundary machinery

## Where the code lives

Main files:

- `rsatb/calc/fftcheb_transport.hpp`
- `rsatb/calc/fft.hpp`
- `rsatb/rsatb.hpp`
- `rsatb/pybind_rsatb.cpp`
- `rsatb/driver.py`
- `rsatb/io/hdf5.hpp`
- `rsatb/model/operator_construction.hpp`

Rough responsibility split:

- `fftcheb_transport.hpp`: the FFTCheb transport algorithm itself
- `fft.hpp`: rowwise FFT/DCT helpers used by the transport code
- `rsatb.hpp`: public C++ API entry points and k-point loop
- `pybind_rsatb.cpp` + `driver.py`: Python bindings and reshaping/unit conversion
- `hdf5.hpp` + `operator_construction.hpp`: complex Hamiltonian input support

## High-level idea

The paper's main speedup comes from avoiding the explicit `M x M` transport moment tensor.

Instead of:

1. generating all Chebyshev moments
2. forming the full double-moment tensor
3. post-processing that tensor into an energy spectrum

the FFTCheb path does this:

1. generate left/right Chebyshev auxiliary sequences in real space
2. sweep those sequences row-by-row
3. transform each row into energy-space vectors with FFT-friendly transforms
4. accumulate the needed rowwise inner products immediately

That changes the expensive post-processing from an `O(D M^2)` tensor build to an `O(D M log M)` row-transform-and-accumulate pattern, where:

- `D` is the local orbital dimension
- `M` is the number of Chebyshev moments / energy points

## RSATB conventions this implementation follows

The transport code does not introduce a second operator convention. It reuses the same non-orthogonal setup already used by `compute_ac`.

Important conventions:

- `Chebyshev` owns the action of the scaled Hamiltonian and, when overlap exists, the `S^-1 H` / adjoint machinery.
- The right transport sequence is built with `current(...)`.
- The left transport sequence is built with `current_adjoint(...)`.
- This means the non-orthogonal velocity placement is the same as the existing AC path.

If you are reading the paper and the code side-by-side, this is the main conceptual bridge:

- the paper writes the formulas in operator form
- `rsatb` realizes the left/right operator placement through `current(...)` and `current_adjoint(...)`

## End-to-end flow

The public entry point is:

- `RSATB::compute_dc_spectrum(...)`

The convenience wrapper is:

- `RSATB::compute_dc(...)`

For each local k-point, `compute_dc_spectrum(...)` does:

1. build `Vx`, `Vy`, `Vz` with `build_overlap_velocity(...)`
2. build a `Chebyshev` object from `H(k)` and, when available, `S(k)`
3. pack the stochastic samples into a `DistMultiVector`
4. call `dc_spectrum_fftcheb(...)`
5. reduce the k-point results across MPI communicators

Inside `dc_spectrum_fftcheb(...)`:

1. loop over stochastic samples
2. call `accumulate_fftcheb_sample(...)` for each sample
3. average over samples
4. MPI-reduce the accumulated energy kernel
5. convert that kernel into the final energy-resolved conductivity

Inside `accumulate_fftcheb_sample(...)`:

1. choose the number of row partitions
2. generate the left/right Chebyshev sequences
3. store only the rows for the active partition
4. transform those rows into energy-space vectors
5. accumulate rowwise inner products for Greenwood or Bastin

## How the rowwise transform works

For each row and axis, the code reconstructs the energy-space vectors from the supplement:

- `phi3`: cosine/Chebyshev transform
- `phi1`: complex phase transform
- `phi2`: same as `phi1`, but weighted by `m`

Implementation detail:

- `phi3` is built with a DCT-style transform using `RowFFTWorkspace::dct_type3(...)`
- `phi1` and `phi2` are built with a zero-padded inverse complex FFT using `RowFFTWorkspace::zero_padded_ifft(...)`

Why the zero padding exists:

- the supplement uses phases evaluated on the Chebyshev nodes
- a length-`2M` inverse FFT gives the needed `exp(i pi m k / M)` structure efficiently

## Greenwood path

The Greenwood branch is the cheaper special case.

For each axis:

1. build `phi3` on the left and right
2. accumulate `conj(phi3_left) * phi3_right / (1 - eps^2)`

Only the diagonal tensor components are filled:

- `xx`
- `yy`
- `zz`

The storage order in the output array is always:

1. `xx`
2. `xy`
3. `xz`
4. `yy`
5. `yz`
6. `zz`

So in Greenwood mode the off-diagonal slots remain zero.

## Bastin path

The Bastin branch follows Eqs. S5-S8 from the supplement.

For each row and axis, the code builds:

- left `phi1`, `phi2`, `phi3`
- right `phi1`, `phi2`, `phi3`

Then for each tensor component `(a, b)` with `a <= b`, it accumulates the rowwise form of Eq. S5:

```text
zeta_ab(eps_k) =
    eps_k * ( <phi_a^L,3 | phi_b^R,1> + <phi_a^L,1 | phi_b^R,3> )
  + i * sqrt(1 - eps_k^2) *
    ( <phi_a^L,2 | phi_b^R,3> - <phi_a^L,3 | phi_b^R,2> )
```

After the row sweep is done, `dc_spectrum_fftcheb(...)` converts the `zeta` kernel into the final zero-temperature conductivity-vs-Fermi-energy curve by cumulative integration on the ascending Chebyshev grid.

This is why the Bastin output should be interpreted as:

- `sigma(mu, T = 0)` sampled on the Chebyshev energy grid

and not just as the raw `zeta(eps)` kernel.

## Partitioning and memory

Large `M` makes the row-storage buffers expensive:

- 3 axes
- left and right sequences
- `row_count x n_moments`
- complex-valued scratch storage

The implementation supports two ways to choose partitioning:

- explicit `partition_sections`
- implicit `max_workspace_mb`

The partitioned algorithm repeats the Chebyshev recurrence for each row section. That costs extra matrix applications, but keeps peak memory bounded because only one row chunk is stored at a time.

This matches the supplement's "partitioned FastCheb" idea.

## Energy grid and output layout

The native energy grid is the Chebyshev node grid:

```text
eps_k = cos(pi * (k + 1/2) / M)
E_k = center + half_width * eps_k
```

Internally many expressions are most natural in the paper's native Chebyshev ordering, but the public output is sorted in ascending physical energy.

Returned shapes:

- C++ `compute_dc_spectrum(...)`
  - `sigma_out`: flattened `[n_k, n_energy, 6]`
  - `energy_out`: `[n_energy]`
- Python `compute_dc_spectrum(...)`
  - `sigma_out.shape == (n_k, n_moments, 6)`
  - `energy_out.shape == (n_moments,)`
- C++ `compute_dc(...)`
  - flattened `[n_k, 6]`
- Python `compute_dc(...)`
  - `sigma_out.shape == (n_k, 6)`

Tensor-component order everywhere:

1. `xx`
2. `xy`
3. `xz`
4. `yy`
5. `yz`
6. `zz`

## Temperature handling

`compute_dc_spectrum(...)` returns the zero-temperature spectrum.

`compute_dc(...)` is a wrapper that integrates that spectrum against `-df/dE`:

```text
sigma(mu, T) = integral dE [ -df(E, mu, T)/dE ] sigma(E, T=0)
```

At `T = 0`, the implementation reduces to interpolation on the Chebyshev grid.

## Complex Hamiltonian support

Hall-type response requires complex Hamiltonians, so the HDF5 path was extended.

Current conventions:

- real Hamiltonian blocks: rank-2 dataset `[m, n]`
- complex Hamiltonian blocks: rank-3 dataset `[m, n, 2]`
  - last dimension stores real/imag parts

The overlap path is still read as real.

`build_from_hdf5(...)` now constructs complex `H` image containers, and the rest of the transport stack works on complex `H(k)` matrices.

## Public API summary

### C++

```cpp
compute_dc_spectrum(
    int n_moments,
    std::vector<double>& sigma_out,
    std::vector<double>& energy_out,
    std::string formulation = "greenwood",
    std::string axis = "abc",
    int partition_sections = 0,
    size_t max_workspace_mb = 0);

compute_dc(
    int n_moments,
    double mu,
    double temp,
    std::vector<double>& dc_out,
    std::string formulation = "greenwood",
    std::string axis = "abc",
    int partition_sections = 0,
    size_t max_workspace_mb = 0);
```

### Python

```python
sigma, energy = rsatb.compute_dc_spectrum(
    n_moments,
    formulation="greenwood",
    axis="abc",
    partition_sections=0,
    max_workspace_mb=0,
)

sigma_mu = rsatb.compute_dc(
    n_moments,
    mu,
    temp,
    formulation="greenwood",
    axis="abc",
    partition_sections=0,
    max_workspace_mb=0,
)
```

Python returns conductivity in `S/m`.

## What to read first if you want to modify this code

Recommended reading order:

1. `doc/formula.md`
2. `doc/random_state.md`
3. `rsatb/calc/ac.hpp`
4. `rsatb/calc/fftcheb_transport.hpp`
5. `rsatb/rsatb.hpp`

Why this order:

- `formula.md` explains the non-orthogonal basis conventions
- `ac.hpp` shows the current/current_adjoint operator placement that FFTCheb reuses
- `fftcheb_transport.hpp` is much easier to follow once those conventions are clear

## Testing that currently covers this path

Relevant tests:

- `rsatb/calc/test/test_fftcheb_transport.cpp`
- `rsatb/io/test.cpp`
- `test/test_binding.py`
- `rsatb/test_rsatb.cpp`

Current coverage includes:

- Greenwood toy-model reference checks
- Bastin toy-model reference checks
- non-orthogonal Greenwood coverage
- partitioned vs unpartitioned equivalence
- batched-sample averaging equivalence
- HDF5 real/complex reader checks
- Python API exposure checks
- regression smoke test for existing DOS/AC paths

## Current limitations and follow-up work

Known limitations:

- two-terminal conductance is not implemented
- there is no separate device/contact abstraction yet

(An earlier limitation -- requiring overlap support and raising when
`S_images_` was absent -- no longer holds: velocity is a model capability,
and orthogonal models get it through the generic commutator builder
(`MakeCommutatorVelocityBuilder`). Both `compute_dc_spectrum` and
`compute_ac` gate only on `has_velocity()`; orthogonal SKTB transport is
covered by the toy-problem GTests and the graphene example.)

Natural next steps:

- add more exact-reference coverage for Hall responses on richer toy models
- add user-facing docs/examples showing how to prepare complex HDF5 inputs
- add a separate conductance module if the repo gains a lead/contact layer

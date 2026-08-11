import _rsatb
import numpy as np
import scipy.signal as sp
from dataclasses import dataclass

# Configuration types are defined once, in C++, and bound directly. Previously
# each field was written four times -- Python dataclass, to_dict(), pybind
# parser, C++ struct -- and an unrecognized key was silently dropped rather
# than raising.
BasisSpec = _rsatb.BasisSpec
Capabilities = _rsatb.Capabilities
HamiltonianFile = _rsatb.HamiltonianFile
LcaoModel = _rsatb.LcaoModel
Model = _rsatb.Model
OrbitalFiles = _rsatb.OrbitalFiles
PreconditionerOptions = _rsatb.PreconditionerOptions
PseudopotentialFiles = _rsatb.PseudopotentialFiles
SetupOptions = _rsatb.SetupOptions
SpinSpec = _rsatb.SpinSpec
Structure = _rsatb.Structure
atomic_number = _rsatb.atomic_number
# Verbosity is process-wide, not a per-calculation argument: the solver
# logs from call sites that never see an RSATB object. Set it before
# building one. How much it says is this; where it goes is the CLI's job.
set_log_level = _rsatb.set_log_level


# The Slater-Koster builders and model types are bound directly from C++.
# The previous Python layer accepted two different input dialects and told
# them apart by inspecting dict key names (_split_onsite, _split_pairs,
# _looks_like_matrix_group); H and S are now separate SkParameters, so there
# is nothing to disambiguate.
DftbModel = _rsatb.DftbModel
SkParameters = _rsatb.SkParameters
SkRadialFunction = _rsatb.SkRadialFunction
SktbModel = _rsatb.SktbModel
TightBindingFileModel = _rsatb.TightBindingFileModel
sk_exp_poly = _rsatb.sk_exp_poly
sk_spline = _rsatb.sk_spline


class RSATB:
    """
    Python wrapper for the RSATB C++ class.
    """

    def __init__(self, structure, model, options=None, npar=1):
        """
        Build a ready-to-compute RSATB from a structure and a model.

        The model reports the basis it works in, so orbital counts and cutoff
        radii are derived rather than supplied.

        Args:
            structure (Structure): Atomic geometry, e.g. Structure.from_file(...).
            model: A model instance such as LcaoModel.
            options (SetupOptions, optional): k-points, sampling, bounds,
                and preconditioner settings.
            npar (int): Number of k-point-parallel sub-communicators.

        Verbosity is not an argument here. It is process-wide state: call
        rsatb.set_log_level('silent'|'summary'|'progress'|'debug') before
        building, so the many solver call sites that never see this object
        still agree on how much to say.
        """
        if options is None:
            options = SetupOptions()
        self._cpp_obj = _rsatb.RSATB(structure, model, options, npar)
        # Velocity and position operators are built lazily through the model,
        # so keep a Python reference alongside pybind's keep_alive.
        self._model = model

    @classmethod
    def from_graph(cls, atom_index, atom_type, edge_index, edge_shift_vec, cell, pos,
                   model, options=None, N_atom=0, atom_offset=0, N_edge=0,
                   npar=1):
        """
        Build an RSATB from a pre-partitioned atomic graph and a model.

        Use this when the neighbour list already exists -- a hand-built model
        cell, or a graph produced by another tool. `type_norb` is not an
        argument: it comes from the model's basis, so the block structure has
        one source of truth.

        This is a *distributed* description: each rank passes only its own
        atoms and edges, and `N_atom` / `N_edge` are the global totals. Handing
        every rank the full graph is only correct when each sub-communicator
        holds a single rank; otherwise the system is duplicated once per rank.
        The local counts are summed and checked against the declared totals, so
        that mistake raises instead of producing a doubled spectrum.

        Args:
            atom_index (array): Global index of each local atom.
            atom_type (array): Atom type index of each local atom.
            edge_index (array): Flattened (src, dst) pairs.
            edge_shift_vec (array): Flattened lattice shift per edge.
            cell (array): Unit cell, 9 values row-major.
            pos (array): Cartesian positions, 3 per atom.
            model: A model instance such as SktbModel.
            options (SetupOptions, optional): Shared setup options.
            N_atom, N_edge (int): Global counts; default to the local counts.
            atom_offset (int): Global offset of this rank's atoms.
            npar (int): Number of k-point-parallel sub-communicators.
        """
        if options is None:
            options = SetupOptions()

        instance = cls.__new__(cls)
        instance._cpp_obj = _rsatb.RSATB(
            np.ascontiguousarray(atom_index, dtype=np.int32),
            np.ascontiguousarray(atom_type, dtype=np.int32),
            np.ascontiguousarray(edge_index, dtype=np.int32),
            np.ascontiguousarray(edge_shift_vec, dtype=np.int32),
            np.ascontiguousarray(cell, dtype=np.float64),
            np.ascontiguousarray(pos, dtype=np.float64),
            model, options, N_atom, atom_offset, N_edge, npar,
        )
        instance._model = model
        return instance

    def test_solve(self):
        self._cpp_obj.test_solve()

    def compute_dos(self, dE, method="kpm"):
        """
        Compute the Density of States (DOS).

        Args:
            dE (float): Target energy resolution in eV. The Chebyshev step
                and moment counts are derived from the spectral span at run
                time (cost is linear in span/dE), so this is the only knob.
            method (str): The broadening kernel; both run on the same moment
                recurrence. "kpm" (Jackson, strictly positive; the default)
                or "corr" (the historical Hann-window + FFT output,
                reconstructed from the moments at the same cost).

        Returns:
            tuple: (dos_out, omega_out) as numpy arrays.
        """
        dos_out, omega_out = self._cpp_obj.compute_dos(dE, method)
        return np.array(dos_out), np.array(omega_out)

    def compute_rs_ldos(self, energies, eta, spacing=0.4, box_edge=0,
                        n_samples=64, block=16, seed=12345):
        """
        Compute the full-cell LDOS map on a real-space grid.

        Square-root-filtered stochastic estimator (Gamma-only; needs a model
        with real-space orbitals, i.e. LCAO or DFTB with .orb files). One
        Chebyshev sweep serves every requested energy, and the map is
        non-negative pointwise by construction.

        Choosing n_samples. With Rademacher draws the pointwise relative
        standard error is

            sigma_rel = sqrt(2 / n_samples)          (n = 2 / sigma^2)

        so 20% needs 50 samples, 10% needs 200, 5% needs 800, and 1% needs
        20000. Measured on diamond sc444 at 0.95x that bound, flat across
        valence, gap and conduction energies. What does NOT appear in it is
        as useful as what does: the error is independent of system size, of
        the energy, of eta, and of the grid spacing -- refining the grid buys
        resolution, never accuracy, and a denser grid is nearly free because
        the Chebyshev sweep does not touch it.

        Any quantity integrated over space converges much faster, because it
        is a Hutchinson trace over the m_eff states inside the eta window:
        roughly sqrt(2 / (n_samples * m_eff)). On sc444 that was a factor of
        ~10 in the bands (m_eff ~ 100-200) and ~4 in the gap (m_eff ~ 15).
        Budget for the map you want to look at, not for the integral.

        Args:
            energies: Evaluation energies in eV, inside the spectral window.
            eta (float): Gaussian broadening in eV; the filter degree scales
                as (spectral span) / eta.
            spacing (float): Grid spacing in Angstrom.
            box_edge (int): Grid box edge in points; 0 picks a physical
                (~2.4 A) default whatever the spacing.
            n_samples (int): Random states; relative noise ~ 1/sqrt(n).
            block (int): States per Chebyshev sweep (SpMM width).
            seed (int): RNG seed, for reproducible maps.

        Returns:
            dict with keys:
                shape: (n0, n1, n2) grid points along the lattice vectors.
                n_point: total grid points.
                point_index: (n_owned,) int64 global index of each owned point.
                ldos: (n_energy, n_owned) values in states/eV/A^3 -- THIS
                    RANK's share of the grid; ranks partition the points.
                grid (serial runs only): ldos scattered to the full
                    (n_energy, n0, n1, n2) array.
                filter_degree, n_states, sampler_residual: diagnostics.
        """
        out = self._cpp_obj.compute_rs_ldos(
            list(energies), eta, spacing, box_edge, n_samples, block, seed)
        if out["ldos"].shape[1] == out["n_point"]:
            n_energy = out["ldos"].shape[0]
            full = np.zeros((n_energy, out["n_point"]))
            full[:, out["point_index"]] = out["ldos"]
            out["grid"] = full.reshape((n_energy,) + tuple(out["shape"]))
        return out

    def energy_bounds(self):
        """
        Return the spectral window (emin, emax) the expansions are scaled to.

        Chebyshev observables are defined on the open interval; energies at or
        beyond the endpoints reconstruct to zero.

        Returns:
            tuple: (emin, emax) in eV.
        """
        return self._cpp_obj.energy_bounds()

    def compute_ac(self, nt, n_kpm, temp, mu, axis="abc", method="concurrent", block=0):
        """
        Compute the AC conductivity.

        Args:
            nt (int): Number of time steps.
            n_kpm (int): Number of KPM moments.
            temp (float): Temperature.
            mu (float): Chemical potential.
            axis (str, optional): Axis for volume calculation. Defaults to "abc".
            method (str): "concurrent" (shared-Chebyshev-state evolution,
                ~9x fewer propagation matvecs; the default) or "sequential"
                (stepwise propagation, the independent cross-check). Both
                sample the correlator at its Nyquist step, alias-free.
            block (int): Concurrent only: time steps per block, bounding the
                stored-state memory. 0 uses the built-in min(nt, 128).

        Returns:
            tuple: (ac_out, omega_out) as numpy arrays.
            unit in S/m for ac_out and cm-1 for omega_out
        """
        ac_out, omega_out = self._cpp_obj.compute_ac(nt, n_kpm, temp, mu, axis, method, block)
        ac_out = np.array(ac_out).reshape(-1, nt+1, 6)
        n_k, ntp1 = ac_out.shape[0], ac_out.shape[1]
        
        sigma = np.zeros((n_k, 2*ntp1, 6))
        sigma[:, ntp1:, :] = ac_out
        sigma[:, 1:ntp1+1, :] = ac_out[:, ::-1, :]
        ac_out = ac_out + 1j * np.imag(sp.hilbert(sigma, axis=1))[:,ntp1:2*ntp1,:]
        # ac_out is now (n_k, nt+1, 6)
        
        # now compute other properties from ac
        # dielectric function
        # epsilon = 1 + i * sigma / (epsilon_0 * omega)
        
        omega_ev = np.array(omega_out)
        omega_out = omega_ev * 8065.56
        epsilon_0 = 8.8541878128e-12
        hbar = 6.582119569e-16 # eV*s
        
        # Avoid division by zero at omega=0
        mask = omega_ev > 1e-10
        eps = np.ones_like(ac_out, dtype=np.complex128)
        
        # sigma is ac_out * 2.434e6
        sigma_si = ac_out * 2.4341348 * 1e6
        
        # eps[:, mask, :] = 1.0 + 1j * sigma_si[:, mask, :] * hbar / (epsilon_0 * omega_ev[mask, None])
        # Need to broadcast omega_ev[mask] to (n_k, sum(mask), 6)
        eps[:, mask, :] = 1.0 + 1j * sigma_si[:, mask, :] * hbar / (epsilon_0 * omega_ev[mask][None, :, None])

        # refractive index N = n + ik = sqrt(epsilon)
        ref_index = np.sqrt(eps)
        n = np.real(ref_index)
        k = np.imag(ref_index)

        # absorption coefficient alpha = 2 * omega_rad_s * k / c
        alpha = np.zeros_like(k)
        alpha[:, mask, :] = 2 * (omega_ev[mask][None, :, None] / hbar) * k[:, mask, :] / 2.99792458e8 / 100.0

        # Reshape everything to (-1, (nt+1)*6) for consistency with previous return format
        ac_out = (ac_out * 2.4341348 * 1e6).reshape(-1, (nt+1)*6)
        eps = eps.reshape(-1, (nt+1)*6)
        n = n.reshape(-1, (nt+1)*6)
        k = k.reshape(-1, (nt+1)*6)
        alpha = alpha.reshape(-1, (nt+1)*6)

        properties = {
            "epsilon": eps,
            "refractive_index": n,
            "extinction_coefficient": k,
            "absorption_coefficient": alpha
        }

        return ac_out, omega_out, properties

    def compute_dc_spectrum(self, n_moments, formulation="greenwood", axis="abc", partition_sections=0, max_workspace_mb=0, kernel="jackson", method="fftcheb"):
        """
        Compute the zero-temperature DC conductivity spectrum on the Chebyshev energy grid.

        This is the direct Python entry point to the FFTCheb transport module.
        The returned spectrum is resolved on the native Chebyshev nodes used by
        the recurrence, so it is a good debugging/analysis surface when working
        on new transport features.

        Args:
            n_moments (int): Number of Chebyshev moments / energy grid points.
            formulation (str, optional): "greenwood" or "bastin".
            axis (str, optional): Axis selector for volume calculation.
            partition_sections (int, optional): Explicit row-partition count.
            max_workspace_mb (int, optional): Workspace cap used to infer row partitioning.
            kernel (str, optional): Chebyshev summation rule: "jackson"
                (default; positive, pointwise-clean spectrum) or "costaper"
                (higher-order taper; recommended when the spectrum is only
                consumed through energy integrals such as compute_dc).

        Returns:
            tuple: (sigma_out, energy_out) with sigma_out shaped (n_k, n_moments, 6)
                   in S/m and energy_out in eV.
                   Tensor-component order is (xx, xy, xz, yy, yz, zz).
        """
        sigma_out, energy_out = self._cpp_obj.compute_dc_spectrum(
            n_moments, formulation, axis, partition_sections, max_workspace_mb, kernel
        )
        sigma_out = np.array(sigma_out, dtype=np.float64).reshape(-1, n_moments, 6)
        sigma_out = sigma_out * 2.4341348 * 1e6
        energy_out = np.array(energy_out, dtype=np.float64)
        return sigma_out, energy_out

    def compute_dc(self, n_moments, mu, temp, formulation="greenwood", axis="abc", partition_sections=0, max_workspace_mb=0, kernel="jackson", method="fftcheb"):
        """
        Evaluate the DC conductivity tensor at a target chemical potential and temperature.

        Internally this reuses `compute_dc_spectrum(...)` and performs the
        thermal convolution on the same Chebyshev energy grid, so it is mainly
        a convenience wrapper rather than a separate transport algorithm.

        Args:
            n_moments (int): Number of Chebyshev moments / energy grid points.
            mu (float): Chemical potential in eV.
            temp (float): Temperature in K.
            formulation (str, optional): "greenwood" or "bastin".
            axis (str, optional): Axis selector for volume calculation.
            partition_sections (int, optional): Explicit row-partition count.
            max_workspace_mb (int, optional): Workspace cap used to infer row partitioning.
            kernel (str, optional): Chebyshev summation rule ("jackson" or
                "costaper"). For this integrated quantity "costaper" gives a
                higher-order bias decay in n_moments; "jackson" remains the
                conservative default.

        Returns:
            ndarray: Conductivity tensor shaped (n_k, 6) in S/m with component
                     order (xx, xy, xz, yy, yz, zz).
        """
        sigma_out = self._cpp_obj.compute_dc(
            n_moments, mu, temp, formulation, axis, partition_sections, max_workspace_mb, kernel
        )
        sigma_out = np.array(sigma_out, dtype=np.float64).reshape(-1, 6)
        sigma_out = sigma_out * 2.4341348 * 1e6
        return sigma_out

    def compute_occupation(self, mu, temp, indicator_threshold=1e-4, n_max=65536):
        """
        Stochastic-trace electron count and band energy at one (mu, T).

        Uses the adaptive tapered (ACE-FD) trace expansion: the Chebyshev
        degree is chosen a posteriori from the sampled spectral measure and,
        on extended systems, is set by the spectral regularity near mu rather
        than by W/kT — i.e. it does not grow as the temperature is lowered.
        No spin-degeneracy factor is applied to the returned counts.

        Args:
            mu (float): Chemical potential in eV.
            temp (float): Temperature in K (0 selects the T=0 step).
            indicator_threshold (float, optional): Level the a-posteriori
                error *indicator* must fall below, relative to the
                sample-vector norm. This is not an error tolerance: no
                inequality between the indicator and the true error is
                proven, and outside the method's stated regime (many levels
                within the resolution window) it can report a fired indicator
                at a degree whose true error is orders of magnitude larger.
                Validate against a fixed-degree run if that regime is in doubt.
            n_max (int, optional): Hard cap on the expansion degree.

        Returns:
            dict: n_electrons, band_energy (eV), mu, n_moments, indicator,
                  indicator_fired. `indicator_fired=False` means the degree
                  cap was reached instead; the result is then Jackson-grade at
                  that degree.
        """
        return self._cpp_obj.compute_occupation(mu, temp, indicator_threshold, n_max)

    def find_fermi_level(self, n_electrons, temp, indicator_threshold=1e-4, n_max=65536,
                         mu_guess=float("nan")):
        """
        Find mu such that the stochastic-trace electron count matches a target.

        One Chebyshev moment set is computed per k-point (matvecs); the whole
        chemical-potential search then runs in coefficient space at zero
        additional matvecs, bracketing with the monotone Jackson summation
        and refining with the tapered one. Temperature scans at fixed
        structure can call this repeatedly at little cost.

        Args:
            n_electrons (float): Target electron count (no spin factor).
            temp (float): Temperature in K.
            indicator_threshold (float, optional): Level the a-posteriori
                error indicator must fall below. See compute_occupation for
                why this is not an error tolerance.
            n_max (int, optional): Hard cap on the expansion degree.
            mu_guess (float, optional): Anchor for the adaptive degree
                selection; defaults to the band center. The indicator is
                re-checked at the returned mu either way.

        Returns:
            dict: n_electrons, band_energy (eV), mu (eV), n_moments,
                  indicator, indicator_fired.
        """
        return self._cpp_obj.find_fermi_level(n_electrons, temp, indicator_threshold, n_max,
                                              mu_guess)

    def compute_msd_spectrum(self, n_steps, dt_fs, n_kpm, axis="x"):
        """
        Compute the mean-square displacement spectrum DeltaX^2(E, t).

        MSD is evaluated from the configured Gamma-point state. The returned
        array therefore always has a singleton leading k-axis.

        Returns:
            tuple: (msd, energy, time)
                msd.shape == (1, n_steps + 1, n_kpm)
                energy.shape == (n_kpm,)
                time.shape == (n_steps + 1,)
        """
        msd_out, energy_out, time_out = self._cpp_obj.compute_msd_spectrum(n_steps, dt_fs, n_kpm, axis)
        msd_out = np.array(msd_out, dtype=np.float64).reshape(-1, n_steps + 1, n_kpm)
        energy_out = np.array(energy_out, dtype=np.float64)
        time_out = np.array(time_out, dtype=np.float64)
        return msd_out, energy_out, time_out

    def compute_diffusion_spectrum(self, n_steps, dt_fs, n_kpm, axis="x"):
        """
        Compute DeltaX^2(E, t) and D(E, t) = 0.5 d_t DeltaX^2(E, t).
        """
        msd, energy, time = self.compute_msd_spectrum(n_steps, dt_fs, n_kpm, axis=axis)
        diffusion = 0.5 * np.gradient(msd, time, axis=1, edge_order=1)
        return msd, diffusion, energy, time

    def fit_ballistic_velocity(self, diffusion, time, fit_window):
        """
        Fit D(E, t) ~ v(E)^2 t in a clean-system ballistic window.

        Args:
            diffusion: array shaped (n_k, n_t, n_energy)
            time: array shaped (n_t,)
            fit_window: (t_min, t_max)

        Returns:
            ndarray shaped (n_k, n_energy) in length / time units of the input geometry/time grid.
        """
        mask = (time >= fit_window[0]) & (time <= fit_window[1])
        if np.count_nonzero(mask) < 2:
            raise ValueError("fit_window must contain at least two time samples")

        t = time[mask]
        t_centered = t - np.mean(t)
        denom = np.sum(t_centered ** 2)
        if denom <= 0.0:
            raise ValueError("fit_window produces a singular linear fit")

        y = diffusion[:, mask, :]
        y_centered = y - np.mean(y, axis=1, keepdims=True)
        slopes = np.tensordot(t_centered, y_centered, axes=(0, 1)) / denom
        return np.sqrt(np.clip(slopes, 0.0, None))

    def extract_mean_free_path(self, diffusion, time, energy, clean_velocity, plateau_window):
        """
        Extract l(E) = 2 Dmax(E) / v_clean(E) from a long-time plateau window.

        Returns:
            tuple: (mfp, dmax) with both shaped like clean_velocity.
        """
        mask = (time >= plateau_window[0]) & (time <= plateau_window[1])
        if np.count_nonzero(mask) < 1:
            raise ValueError("plateau_window must contain at least one time sample")

        dmax = np.mean(diffusion[:, mask, :], axis=1)
        safe_velocity = np.where(np.abs(clean_velocity) > 1e-12, clean_velocity, np.nan)
        mfp = 2.0 * dmax / safe_velocity
        return mfp, dmax

    def compute_quantum_metric(self, nt, n_kpm, temp, mu, axis="abc", omega_min_ev=0.0):
        """
        Compute the longitudinal quantum metric Gxx from the existing AC conductivity.

        Returns:
            tuple: (gxx, sigma_xx, omega_ev)
                gxx.shape == (n_k,)
                sigma_xx.shape == (n_k, nt + 1)
                omega_ev.shape == (nt + 1,)
        """
        gxx, sigma_xx, omega_ev = self._cpp_obj.compute_quantum_metric(
            nt, n_kpm, temp, mu, axis, omega_min_ev
        )
        gxx = np.array(gxx, dtype=np.float64)
        sigma_xx = np.array(sigma_xx, dtype=np.float64).reshape(-1, nt + 1)
        omega_ev = np.array(omega_ev, dtype=np.float64)
        return gxx, sigma_xx, omega_ev

    def get_kernel_info(self):
        """
        Get information about the computational kernel.

        Returns:
            str: Kernel information.
        """
        return self._cpp_obj.get_kernel_info()

    def atomic_data(self):
        """
        Get the atomic data object.
        """
        return self._cpp_obj.atomic_data()

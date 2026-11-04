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
        """Build an RSATB object from a partitioned atomic graph and a model.

        Use this when the neighbour list already exists, for example for a model
        cell built by hand or a graph produced by another program. The orbital
        counts are taken from the model's basis.

        Each MPI process passes only its own atoms and edges; `N_atom` and
        `N_edge` are the totals over all processes. The local counts are summed
        and checked against these totals, so passing the full graph to every
        process raises an error instead of duplicating the system.

        Args:
            atom_index (array): Global index of each local atom.
            atom_type (array): Atom type index of each local atom.
            edge_index (array): Flattened (src, dst) pairs.
            edge_shift_vec (array): Flattened lattice shift per edge.
            cell (array): Unit cell, 9 values, row-major.
            pos (array): Cartesian positions, 3 per atom.
            model: A model object such as SktbModel.
            options (SetupOptions, optional): Setup options.
            N_atom, N_edge (int): Total counts; default to the local counts.
            atom_offset (int): Global offset of this process's atoms.
            npar (int): Number of k-point groups.
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
        """Compute the density of states.

        Args:
            dE (float): Energy resolution in eV. The number of Chebyshev moments
                follows from the energy window; the cost is proportional to
                (emax - emin) / dE.
            method (str): "kpm" (default): Jackson kernel; the result is
                positive. "corr": a Hann window on the time correlation function
                reconstructed from the same moments.

        Returns:
            tuple: (dos, energy) as NumPy arrays. `dos` is in states/eV of the
            simulated cell, both spins counted: it integrates to n_states() over
            the whole spectrum and to the electron count up to the Fermi level.
            It holds all k-points in one flat array;
            `dos.reshape(-1, len(energy))` separates them.
        """
        dos_out, omega_out = self._cpp_obj.compute_dos(dE, method)
        return np.array(dos_out), np.array(omega_out)

    def compute_pdos(self, dE, projection="atom"):
        """Compute the density of states projected onto groups of orbitals.

        Each group costs about as much as one compute_dos call, and the groups
        sum to the total DOS, which is returned as 'total'.

        The projection follows the operator form: the sandwich form projects
        onto Lowdin orbitals (non-negative), the solve form gives Mulliken
        populations, which can be negative. 'population' records which was used.

        Args:
            dE (float): Energy resolution in eV, as in compute_dos.
            projection (str): 'atom': one group per atom, in input order.
                'orbital_slot': one group per element and orbital index within
                the atom; 'group_type' (atomic number, increasing) and
                'group_slot' (orbital index in the basis order) name the groups.

        Returns:
            dict with keys:
                energy: (2*nt,) energy grid in eV.
                pdos: (n_kpoint, n_group, 2*nt) projected DOS, in states/eV as
                    compute_dos.
                total: (n_kpoint, 2*nt) total DOS; equals pdos.sum(axis=1) up to
                    rounding.
                n_group: number of groups.
                population: 'lowdin' or 'mulliken'.
                group_type, group_slot: (n_group,) for projection 'orbital_slot';
                    empty for 'atom'.
        """
        return self._cpp_obj.compute_pdos(dE, projection)

    def compute_rs_density(self, mu, temp, spin_degeneracy=2.0, spacing=0.4,
                           box_edge=0, n_samples=64, block=16, seed=12345,
                           fragment_core=4.5, fragment_buffer=0.0,
                           energy_windows=1, fragment_coefficient="fitted",
                           fragment_workspace_mb=0):
        """Compute the electron density on a real-space grid.

        The density is computed for the given Hamiltonian at chemical potential
        mu and temperature temp; it is not a self-consistent density. Requires a
        single k-point at Gamma and a model with orbital files (lcao).

        The cost is proportional to (emax - emin) / temp, so low temperatures
        are expensive. For an insulator the density does not depend on the
        temperature while it is small compared with the band gap.

        Args:
            mu (float): Chemical potential in eV (see find_fermi_level).
            temp (float): Temperature of the Fermi-Dirac occupation in K.
            spin_degeneracy (float): Electrons per state; 2 for a spin-degenerate
                basis.
            spacing, box_edge, n_samples, block, seed: as in compute_rs_ldos.
            fragment_core, fragment_buffer (float): Fragment correction, in
                Angstrom; fragment_buffer > 0 enables it. A deterministic
                calculation on local fragments gives most of the density, and the
                random vectors correct the remainder. A larger buffer makes the
                fragment part more accurate; fragment_core affects only the cost
                of the fragment calculations.
            energy_windows (int): Number of energy windows the occupation is
                divided into; each is sampled separately in the same Chebyshev
                sweep. This reduces the noise for metals, where the fragment
                correction does not help. 1 disables it.
            fragment_coefficient (str): "fixed": coefficient 1, which keeps the
                estimate unbiased. "fitted": estimated from the same random
                vectors.

        Returns:
            dict with keys:
                shape, n_point, point_index: as in compute_rs_ldos.
                density: (n_owned,) electrons per cubic Angstrom at the grid points
                    of this process.
                grid (serial runs only): the density on the full (n0, n1, n2) grid.
                electrons_on_grid: the density integrated over the grid; it
                    approaches the electron count as the grid is refined.
                filter_degree, n_states, sampler_residual: diagnostics.
        """
        kT = 8.617333262e-5 * temp  # eV/K
        if fragment_coefficient not in ("fixed", "fitted"):
            raise ValueError("fragment_coefficient must be 'fixed' or 'fitted'")
        out = self._cpp_obj.compute_rs_density(
            mu, kT, spin_degeneracy, spacing, box_edge, n_samples, block, seed,
            fragment_core, fragment_buffer, energy_windows,
            fragment_coefficient == "fitted", False, fragment_workspace_mb)
        if out["density"].shape[0] == out["n_point"]:
            full = np.zeros(out["n_point"])
            full[out["point_index"]] = out["density"]
            out["grid"] = full.reshape(tuple(out["shape"]))
        return out

    def compute_rs_ldos(self, energies, eta, spacing=0.4, box_edge=0,
                        n_samples=64, block=16, seed=12345, sketch_families=6):
        """Compute the local density of states on a real-space grid.

        Requires a single k-point at Gamma and a model with orbital files
        (lcao). All energies are computed in one Chebyshev sweep. The estimate
        is unbiased; its relative error is sqrt(2 (1 - c) / k), where k =
        sketch_families * n_samples and c is the captured fraction of the energy
        window, and it is exact once k reaches the number of states in the
        window. The method is described in the documentation under Theory,
        Random-state traces.

        Args:
            energies (list of float): Energies in eV (absolute).
            eta (float): Gaussian broadening (standard deviation), eV.
            spacing (float): Grid spacing, Angstrom.
            box_edge (int): Grid box side in points; 0 selects boxes about
                2.4 Angstrom wide.
            n_samples (int): Number of random vectors.
            block (int): Random vectors per Chebyshev sweep.
            seed (int): Random seed. The random vectors are tied to the atoms,
                not to the MPI processes, so the result does not depend on the
                number of processes apart from rounding.
            sketch_families (int): Chebyshev filter families per random vector.
                Families of one vector are correlated, so keep the default (6)
                and increase n_samples for accuracy.

        Returns:
            dict with keys:
                shape: (n0, n1, n2) grid points along the lattice vectors.
                n_point: total number of grid points.
                point_index: (n_owned,) global index of each grid point of this
                    process.
                ldos: (n_energy, n_owned) values in states/eV/Angstrom^3 at the
                    grid points of this process.
                grid (serial runs only): the LDOS on the full (n_energy, n0, n1, n2)
                    grid.
                capture, sketch_rank, predicted_rel_error: per energy, the
                    captured fraction c (1 = exact), the rank k, and
                    sqrt(2 (1 - c) / k).
                filter_degree, n_states, sampler_residual: diagnostics.
        """
        out = self._cpp_obj.compute_rs_ldos(
            list(energies), eta, spacing, box_edge, n_samples, block, seed,
            sketch_families)
        if out["ldos"].shape[1] == out["n_point"]:
            n_energy = out["ldos"].shape[0]
            full = np.zeros((n_energy, out["n_point"]))
            full[:, out["point_index"]] = out["ldos"]
            out["grid"] = full.reshape((n_energy,) + tuple(out["shape"]))
        return out

    def n_states(self):
        """Return the states in the simulated cell, both spins counted.

        The DOS integrates to this over the whole spectrum: twice the basis size
        for a spin-degenerate basis, the basis size for a spinor one.
        """
        return self._cpp_obj.n_states()

    def energy_bounds(self):
        """Return the energy window (emin, emax) that the expansions are scaled to.

        Results at or beyond the ends of the window are zero.

        Returns:
            tuple: (emin, emax) in eV.
        """
        return self._cpp_obj.energy_bounds()

    def compute_ac(self, nt, n_kpm, temp, mu, axis="abc", method="concurrent", block=0):
        """Compute the optical (AC) conductivity.

        Args:
            nt (int): Number of time steps.
            n_kpm (int): Number of Chebyshev moments for the Fermi-Dirac
                occupation.
            temp (float): Temperature in K.
            mu (float): Chemical potential in eV.
            axis (str, optional): Cell axes that define the normalization volume:
                "abc" (cell volume), "ab"/"bc"/"ca" (area of a lattice plane),
                "a"/"b"/"c" (length of a lattice vector). Default "abc".
            method (str): "concurrent" (default): time steps share Chebyshev
                states, which needs fewer matrix-vector products. "sequential":
                one step at a time; used as an independent check.
            block (int): For "concurrent": time steps per block, which limits
                memory. 0 uses min(nt, 128).

        Returns:
            tuple: (sigma, omega). sigma in S/m, omega in cm^-1. Both spins are
            included: a spin-degenerate basis is counted twice, a spinor basis
            already holds both.
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
        """Compute the DC conductivity at zero temperature as a function of energy.

        The spectrum is given on the Chebyshev energy nodes.

        Args:
            n_moments (int): Number of Chebyshev moments, which is also the
                number of energy points.
            formulation (str, optional): "greenwood" or "bastin".
            axis (str, optional): Cell axes that define the normalization volume,
                as in compute_ac.
            partition_sections (int, optional): Number of row blocks, to limit
                memory; 0 chooses it from max_workspace_mb.
            max_workspace_mb (int, optional): Memory limit in MB used to choose
                partition_sections; 0 means no limit.
            kernel (str, optional): "jackson" (default; the result is positive)
                or "costaper" (smaller bias; better when the spectrum is only
                used inside an energy integral, as in compute_dc).

        Returns:
            tuple: (sigma, energy). sigma has shape (n_k, n_moments, 6), in S/m,
            with both spins included and components in the order
            (xx, xy, xz, yy, yz, zz); energy is in eV.
        """
        sigma_out, energy_out = self._cpp_obj.compute_dc_spectrum(
            n_moments, formulation, axis, partition_sections, max_workspace_mb, kernel
        )
        sigma_out = np.array(sigma_out, dtype=np.float64).reshape(-1, n_moments, 6)
        sigma_out = sigma_out * 2.4341348 * 1e6
        energy_out = np.array(energy_out, dtype=np.float64)
        return sigma_out, energy_out

    def compute_dc(self, n_moments, mu, temp, formulation="greenwood", axis="abc", partition_sections=0, max_workspace_mb=0, kernel="jackson", method="fftcheb"):
        """Compute the DC conductivity tensor at one chemical potential and temperature.

        This computes the spectrum of compute_dc_spectrum and integrates it with
        the derivative of the Fermi-Dirac function.

        Args:
            n_moments (int): Number of Chebyshev moments.
            mu (float): Chemical potential in eV.
            temp (float): Temperature in K.
            formulation (str, optional): "greenwood" or "bastin".
            axis (str, optional): Cell axes that define the normalization volume,
                as in compute_ac.
            partition_sections (int, optional): Number of row blocks, as in
                compute_dc_spectrum.
            max_workspace_mb (int, optional): Memory limit in MB, as in
                compute_dc_spectrum.
            kernel (str, optional): "jackson" (default) or "costaper", whose bias
                decreases faster with n_moments.

        Returns:
            ndarray: Conductivity tensor of shape (n_k, 6) in S/m, with both spins
            included and components in the order (xx, xy, xz, yy, yz, zz).
        """
        sigma_out = self._cpp_obj.compute_dc(
            n_moments, mu, temp, formulation, axis, partition_sections, max_workspace_mb, kernel
        )
        sigma_out = np.array(sigma_out, dtype=np.float64).reshape(-1, 6)
        sigma_out = sigma_out * 2.4341348 * 1e6
        return sigma_out

    def compute_occupation(self, mu, temp, indicator_threshold=1e-4, n_max=65536):
        """Compute the electron count and band energy at one chemical potential and temperature.

        The expansion degree is chosen after the moments are computed, from the
        sampled spectrum. For extended systems it is set by the smoothness of the
        DOS near mu, not by the temperature, so it does not grow as the
        temperature is lowered.

        Args:
            mu (float): Chemical potential in eV.
            temp (float): Temperature in K; 0 gives a step function.
            indicator_threshold (float, optional): Threshold of the error
                indicator that chooses the degree. The indicator is not an error
                bound. For systems with few states near mu, check the result with
                a larger n_max.
            n_max (int, optional): Maximum expansion degree.

        Returns:
            dict: n_electrons (of the simulated cell, both spins counted),
            band_energy (eV), mu, n_moments, indicator, indicator_fired.
            indicator_fired is False when n_max was reached
            before the indicator fell below its threshold.
        """
        return self._cpp_obj.compute_occupation(mu, temp, indicator_threshold, n_max)

    def find_fermi_level(self, n_electrons, temp, indicator_threshold=1e-4, n_max=65536,
                         mu_guess=float("nan")):
        """Find the chemical potential at which the electron count equals a target.

        One set of Chebyshev moments is computed per k-point; the search then
        reuses them without further matrix-vector products, so repeated calls,
        for example at several temperatures, cost little.

        Args:
            n_electrons (float): Target electron count of the simulated cell,
                both spins counted.
            temp (float): Temperature in K.
            indicator_threshold (float, optional): Threshold of the error
                indicator; see compute_occupation.
            n_max (int, optional): Maximum expansion degree.
            mu_guess (float, optional): Energy at which the expansion degree is
                chosen; default the centre of the energy window. The indicator is
                checked again at the returned mu.

        Returns:
            dict: n_electrons, band_energy (eV), mu (eV), n_moments, indicator,
            indicator_fired.
        """
        return self._cpp_obj.find_fermi_level(n_electrons, temp, indicator_threshold, n_max,
                                              mu_guess)

    def compute_msd_spectrum(self, n_steps, dt_fs, n_kpm, axis="x", method="auto",
                             n_kpm_compute=0):
        """Compute the mean-square displacement DeltaX^2(E, t).

        The MSD is computed at the Gamma point, so the returned array has a
        leading k axis of length one. The model must provide the velocity
        operator.

        Args:
            method (str): Both methods compute the same quantity. "stepwise"
                expands the energy resolution again at each output time; it
                suits orthogonal models, whose matrix-vector products are cheap.
                "spectral" uses one Chebyshev chain whose cost does not depend
                on the number of output times; it suits non-orthogonal models.
                "auto" (default) chooses between them; set the method explicitly
                when results of several runs are compared.
            n_kpm_compute (int): 0 disables it. Compute this many Chebyshev
                moments and extend them to n_kpm by linear prediction. Not
                reliable for small cells with discrete spectra, where it can give
                negative values; a warning is printed when it is set.

        Returns:
            tuple: (msd, energy, time)
                msd.shape == (1, n_steps + 1, n_kpm)
                energy.shape == (n_kpm,)
                time.shape == (n_steps + 1,)
        """
        msd_out, energy_out, time_out = self._cpp_obj.compute_msd_spectrum(
            n_steps, dt_fs, n_kpm, axis, method, n_kpm_compute)
        msd_out = np.array(msd_out, dtype=np.float64).reshape(-1, n_steps + 1, n_kpm)
        energy_out = np.array(energy_out, dtype=np.float64)
        time_out = np.array(time_out, dtype=np.float64)
        return msd_out, energy_out, time_out

    def compute_diffusion_spectrum(self, n_steps, dt_fs, n_kpm, axis="x", method="auto",
                                   n_kpm_compute=0):
        """Compute DeltaX^2(E, t) and D(E, t) = 0.5 d/dt DeltaX^2(E, t).

        The arguments are those of compute_msd_spectrum.
        """
        msd, energy, time = self.compute_msd_spectrum(
            n_steps, dt_fs, n_kpm, axis=axis, method=method, n_kpm_compute=n_kpm_compute)
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
        """Return a description of the compiled kernel.

        Returns:
            str: Kernel information.
        """
        return self._cpp_obj.get_kernel_info()

    def atomic_data(self):
        """Return the atomic data object."""
        return self._cpp_obj.atomic_data()

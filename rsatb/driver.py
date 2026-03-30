import _rsatb
import numpy as np
import scipy.signal as sp

class RSATB:
    """
    Python wrapper for the RSATB C++ class.
    """
    def __init__(self, cpp_obj):
        """
        Private constructor. Use classmethods to initialize.
        """
        self._cpp_obj = cpp_obj

    @classmethod
    def from_file(cls, npar, filename, r_max_per_type, type_norb, verbose=True):
        """
        Initialize RSATB from a file.

        Args:
            npar (int): Number of parallel processes.
            filename (str): Path to the structure file.
            r_max_per_type (list of float): Cutoff radii per atom type.
            type_norb (list of int): Number of orbitals per atom type.
        """
        cpp_obj = _rsatb.RSATB(npar, filename, r_max_per_type, type_norb, verbose)
        return cls(cpp_obj)

    @classmethod
    def from_points(cls, npar, pos, z, cell, pbc, r_max_per_type, type_norb, verbose=True):
        """
        Initialize RSATB from atomic positions.

        Args:
            npar (int): Number of parallel processes.
            pos (list of float): Atomic positions.
            z (list of int): Atomic numbers.
            cell (list of float): Unit cell vectors.
            pbc (list of bool): Periodic boundary conditions.
            r_max_per_type (list of float): Cutoff radii per atom type.
            type_norb (list of int): Number of orbitals per atom type.
        """
        cpp_obj = _rsatb.RSATB(npar, pos, z, cell, pbc, r_max_per_type, type_norb, verbose)
        return cls(cpp_obj)

    @classmethod
    def from_raw_data(cls, npar, n_atom, N_atom, atom_offset, n_edge, N_edge, 
                      atom_index, atom_type, edge_index, type_norb, edge_shift_vec, 
                      cell, pos, verbose=True):
        """
        Initialize RSATB from raw arrays (matching the pointer-based C++ constructor).

        Args:
            npar (int): Number of parallel processes.
            n_atom (int): Number of local atoms.
            N_atom (int): Total number of atoms.
            atom_offset (int): Offset of local atoms.
            n_edge (int): Number of local edges.
            N_edge (int): Total number of edges.
            atom_index (array): Indices of atoms.
            atom_type (array): Types of atoms.
            edge_index (array): Indices of edges.
            type_norb (array): Number of orbitals per type.
            edge_shift_vec (array): Shift vectors for edges.
            cell (array): Unit cell.
            pos (array): Atomic positions.
        """
        atom_index = np.ascontiguousarray(atom_index, dtype=np.int32)
        atom_type = np.ascontiguousarray(atom_type, dtype=np.int32)
        edge_index = np.ascontiguousarray(edge_index, dtype=np.int32)
        type_norb = np.ascontiguousarray(type_norb, dtype=np.int32)
        edge_shift_vec = np.ascontiguousarray(edge_shift_vec, dtype=np.int32)
        cell = np.ascontiguousarray(cell, dtype=np.float64)
        pos = np.ascontiguousarray(pos, dtype=np.float64)
        
        cpp_obj = _rsatb.RSATB(npar, n_atom, N_atom, atom_offset, n_edge, N_edge,
                               atom_index, atom_type, edge_index, type_norb, edge_shift_vec,
                               cell, pos, verbose)
        return cls(cpp_obj)

    def setup(self, Hfile, kpoints, orb_names=None, orb_dirs="", psd_names=None, psd_dirs="", tol=1e-7, sptol=1e-5, maxiter=20, method="graph", precond="H/S", nsamples=1):
        """
        Setup the RSATB calculation.

        Args:
            Hfile (str): Path to the Hamiltonian file.
            kpoints (list of list of float): K-points for the calculation.
            orb_names (list of str, optional): Names of the orbitals.
            orb_dirs (str, optional): Directory containing orbital files.
            psd_names (list of str, optional): Names of the pseudopotentials.
            psd_dirs (str, optional): Directory containing pseudopotential files.
            precond (str, optional): Preconditioner type. Defaults to "H/S".
            nsamples (int, optional): Number of samples for random state method. Defaults to 1.
        """
        if orb_names is None: orb_names = []
        if psd_names is None: psd_names = []
        self._cpp_obj.setup(Hfile, kpoints, orb_names, orb_dirs, psd_names, psd_dirs, tol, sptol, maxiter, method, precond, nsamples)

    def setup_config(self, config):
        """
        Structured setup entry point for backend/operator options.

        Expected top-level keys include:
            kpoints, nsamples, compute_bounds, sources, preconditioner, disorder
        """
        self._cpp_obj.setup_config(config)

    def test_solve(self):
        self._cpp_obj.test_solve()

    def compute_dos(self, nt):
        """
        Compute the Density of States (DOS).

        Args:
            nt (int): Number of time steps.

        Returns:
            tuple: (dos_out, omega_out) as numpy arrays.
        """
        dos_out, omega_out = self._cpp_obj.compute_dos(nt)
        return np.array(dos_out), np.array(omega_out)

    def compute_ac(self, nt, n_kpm, temp, mu, axis="abc"):
        """
        Compute the AC conductivity.

        Args:
            nt (int): Number of time steps.
            n_kpm (int): Number of KPM moments.
            temp (float): Temperature.
            mu (float): Chemical potential.
            axis (str, optional): Axis for volume calculation. Defaults to "abc".

        Returns:
            tuple: (ac_out, omega_out) as numpy arrays.
            unit in S/m for ac_out and cm-1 for omega_out
        """
        ac_out, omega_out = self._cpp_obj.compute_ac(nt, n_kpm, temp, mu, axis)
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

    def compute_dc_spectrum(self, n_moments, formulation="greenwood", axis="abc", partition_sections=0, max_workspace_mb=0):
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

        Returns:
            tuple: (sigma_out, energy_out) with sigma_out shaped (n_k, n_moments, 6)
                   in S/m and energy_out in eV.
                   Tensor-component order is (xx, xy, xz, yy, yz, zz).
        """
        sigma_out, energy_out = self._cpp_obj.compute_dc_spectrum(
            n_moments, formulation, axis, partition_sections, max_workspace_mb
        )
        sigma_out = np.array(sigma_out, dtype=np.float64).reshape(-1, n_moments, 6)
        sigma_out = sigma_out * 2.4341348 * 1e6
        energy_out = np.array(energy_out, dtype=np.float64)
        return sigma_out, energy_out

    def compute_dc(self, n_moments, mu, temp, formulation="greenwood", axis="abc", partition_sections=0, max_workspace_mb=0):
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

        Returns:
            ndarray: Conductivity tensor shaped (n_k, 6) in S/m with component
                     order (xx, xy, xz, yy, yz, zz).
        """
        sigma_out = self._cpp_obj.compute_dc(
            n_moments, mu, temp, formulation, axis, partition_sections, max_workspace_mb
        )
        sigma_out = np.array(sigma_out, dtype=np.float64).reshape(-1, 6)
        sigma_out = sigma_out * 2.4341348 * 1e6
        return sigma_out

    def compute_msd_spectrum(self, n_steps, dt_fs, n_kpm, axis="x"):
        """
        Compute the mean-square displacement spectrum DeltaX^2(E, t).

        Returns:
            tuple: (msd, energy, time)
                msd.shape == (n_k, n_steps + 1, n_kpm)
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

    def get_operator_construction(self):
        """
        Deprecated compatibility shim. The new backend exposes an OperatorSet
        in C++, but that object is not yet bound into Python.
        """
        raise AttributeError(
            "OperatorConstruction is deprecated. The new OperatorSet accessor is currently only available in C++."
        )

    def atomic_data(self):
        """
        Get the atomic data object.
        """
        return self._cpp_obj.atomic_data()

import _rsatb
import numpy as np
import scipy.signal as sp
from dataclasses import dataclass, field


def _drop_none_values(data):
    return {key: value for key, value in data.items() if value is not None}


@dataclass
class PreconditionerOptions:
    mode: str = "H/S"
    tol: float = 1e-7
    sptol: float = 1e-5
    maxiter: int = 20
    method: str = "graph"

    def to_dict(self):
        return {
            "mode": self.mode,
            "tol": self.tol,
            "sptol": self.sptol,
            "maxiter": self.maxiter,
            "method": self.method,
        }


@dataclass
class SetupOptions:
    kpoints: list = field(default_factory=lambda: [[0.0, 0.0, 0.0]])
    nsamples: int = 1
    compute_bounds: bool = True
    preconditioner: PreconditionerOptions = field(default_factory=PreconditionerOptions)

    def to_dict(self):
        return {
            "kpoints": self.kpoints,
            "nsamples": self.nsamples,
            "compute_bounds": self.compute_bounds,
            "preconditioner": self.preconditioner.to_dict(),
        }


@dataclass
class Hdf5Input:
    filename: str
    lspinorb: bool = False
    nspin: int = 1
    noncolin: bool = False

    def to_dict(self):
        return {
            "filename": self.filename,
            "lspinorb": self.lspinorb,
            "nspin": self.nspin,
            "noncolin": self.noncolin,
        }


@dataclass
class AtomicSpec:
    type_symbols: list
    type_norb: list
    r_max_per_type: list


def _pairwise_atomic_spec_from_prepared(prepared):
    return AtomicSpec(
        type_symbols=list(prepared["type_symbols"]),
        type_norb=list(prepared["type_norb"]),
        r_max_per_type=list(prepared["r_max_per_type"]),
    )


def _require_pairwise_native_config(kind, native):
    if not native:
        raise ValueError(f"{kind} atomic preparation requires a native pairwise model definition.")
    return native


def sk_spline(r, values, r_cut=None):
    distances = [float(value) for value in r]
    radial_values = [float(value) for value in values]
    if not distances:
        raise ValueError("sk_spline requires at least one radial distance.")
    if len(distances) != len(radial_values):
        raise ValueError("sk_spline requires equally sized r and values arrays.")
    return {
        "kind": "spline",
        "r": distances,
        "values": radial_values,
        "r_cut": float(max(distances) if r_cut is None else r_cut),
    }


def sk_exp_poly(coeffs, r_cut, lambda_value=0.0, r_on=0.0):
    return {
        "kind": "exp_poly",
        "coeffs": [float(value) for value in coeffs],
        "lambda": float(lambda_value),
        "r_on": float(r_on),
        "r_cut": float(r_cut),
    }


def _matrix_key(value):
    key = str(value).strip().lower()
    if key in ("h", "hamiltonian"):
        return "H"
    if key in ("s", "overlap"):
        return "S"
    return None


def _radial_config(value):
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return dict(value)


def _pair_species(value):
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace("->", "-").split("-") if part.strip()]
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError("SKTB pair keys must contain exactly two species labels.")
    return [str(parts[0]), str(parts[1])]


def _channel_name(shell_pair, channel):
    shell = str(shell_pair).strip().lower().replace("-", "").replace("_", "")
    name = str(channel).strip().lower()
    return f"{shell}_{name}"


def _looks_like_matrix_group(data):
    return bool(data) and all(_matrix_key(key) is not None for key in data.keys())


def _split_onsite(onsite):
    if onsite is None:
        return {}, None
    onsite = dict(onsite)
    if _looks_like_matrix_group(onsite):
        result = {"H": {}, "S": None}
        for key, value in onsite.items():
            result[_matrix_key(key)] = dict(value)
        return result["H"], result["S"]
    return onsite, None


def _lower_pair_channels(shells):
    channels = {}
    for shell_pair, shell_channels in dict(shells).items():
        for channel, radial in dict(shell_channels).items():
            channels[_channel_name(shell_pair, channel)] = _radial_config(radial)
    return channels


def _split_pairs(pairs):
    result = {"H": [], "S": []}
    if pairs is None:
        return result
    if isinstance(pairs, list):
        result["H"] = list(pairs)
        return result

    for species_key, pair_data in dict(pairs).items():
        species = _pair_species(species_key)
        pair_data = dict(pair_data)
        matrix_groups = pair_data if _looks_like_matrix_group(pair_data) else {"H": pair_data}
        for key, shells in matrix_groups.items():
            matrix = _matrix_key(key)
            if matrix is None:
                raise ValueError(f"Unsupported SKTB pair matrix key: {key}")
            channels = _lower_pair_channels(shells)
            if channels:
                result[matrix].append({"species": species, "channels": channels})
    return result


def _build_sktb_native_config(basis, hamiltonian=None, overlap=None, onsite=None, pairs=None):
    if hamiltonian is not None and (onsite is not None or pairs is not None):
        raise ValueError("Use either legacy hamiltonian/overlap input or new onsite/pairs input, not both.")
    if hamiltonian is not None:
        native = {"basis": basis, "hamiltonian": hamiltonian}
        if overlap is not None:
            native["overlap"] = overlap
        return native

    h_onsite, s_onsite = _split_onsite(onsite)
    split_pairs = _split_pairs(pairs)
    native = {
        "basis": basis,
        "hamiltonian": {
            "onsite": h_onsite,
            "pairs": split_pairs["H"],
        },
    }
    if s_onsite is not None or split_pairs["S"]:
        native["overlap"] = {
            "onsite": {} if s_onsite is None else s_onsite,
            "pairs": split_pairs["S"],
        }
    if overlap is not None:
        native["overlap"] = overlap
    return native


@dataclass
class LCAOIntegralInput:
    orb_names: list = field(default_factory=list)
    orb_dir: str = ""
    psd_names: list = field(default_factory=list)
    psd_dir: str = ""
    nspin: int = 1
    lspinorb: bool = False
    noncolin: bool = False
    log_file: str = "op2c.log"

    def to_dict(self):
        return {
            "orb_names": list(self.orb_names),
            "orb_dir": self.orb_dir,
            "psd_names": list(self.psd_names),
            "psd_dir": self.psd_dir,
            "nspin": self.nspin,
            "lspinorb": self.lspinorb,
            "noncolin": self.noncolin,
            "log_file": self.log_file,
        }


class LCAOModel:
    def __init__(self, input=None, integrals=None):
        self.input = input
        self.integrals = integrals if integrals is not None else LCAOIntegralInput()

    @classmethod
    def hdf5(cls, filename, integrals=None, **kwargs):
        return cls(input=Hdf5Input(filename, **kwargs), integrals=integrals)

    def to_dict(self):
        result = {"kind": "lcao"}
        if self.input is not None:
            result["input"] = self.input.to_dict()
        if self.integrals is not None:
            result["integrals"] = self.integrals.to_dict()
        return result


class SKTBModel:
    def __init__(self, native=None, input=None):
        self.native = {} if native is None else dict(native)
        self.input = input

    @classmethod
    def native_model(cls, basis, hamiltonian=None, overlap=None, input=None, onsite=None, pairs=None):
        native = _build_sktb_native_config(
            basis=basis,
            hamiltonian=hamiltonian,
            overlap=overlap,
            onsite=onsite,
            pairs=pairs,
        )
        return cls(native=native, input=input)

    native = native_model

    @classmethod
    def hdf5(cls, filename, native=None, **kwargs):
        return cls(native=native, input=Hdf5Input(filename, **kwargs))

    def prepare_atomic_spec_from_file(self, filename):
        native = _require_pairwise_native_config("SKTB", self.native)
        prepared = _rsatb._prepare_sktb_atomic_spec_from_file(filename, native)
        return _pairwise_atomic_spec_from_prepared(prepared)

    def prepare_atomic_spec_from_atomic_numbers(self, z):
        native = _require_pairwise_native_config("SKTB", self.native)
        prepared = _rsatb._prepare_sktb_atomic_spec_from_atomic_numbers(z, native)
        return _pairwise_atomic_spec_from_prepared(prepared)

    def to_dict(self):
        result = {"kind": "sktb"}
        if self.input is not None:
            result["input"] = self.input.to_dict()
        if self.native:
            result["native"] = self.native
        return result


class DFTBModel:
    def __init__(self, native=None, input=None):
        self.native = {} if native is None else dict(native)
        self.input = input

    @classmethod
    def native_model(cls, basis, hamiltonian, overlap, input=None):
        native = {
            "basis": basis,
            "hamiltonian": hamiltonian,
            "overlap": overlap,
        }
        return cls(native=native, input=input)

    native = native_model

    @classmethod
    def hdf5(cls, filename, native=None, **kwargs):
        return cls(native=native, input=Hdf5Input(filename, **kwargs))

    def prepare_atomic_spec_from_file(self, filename):
        native = _require_pairwise_native_config("DFTB", self.native)
        prepared = _rsatb._prepare_sktb_atomic_spec_from_file(filename, native)
        return _pairwise_atomic_spec_from_prepared(prepared)

    def prepare_atomic_spec_from_atomic_numbers(self, z):
        native = _require_pairwise_native_config("DFTB", self.native)
        prepared = _rsatb._prepare_sktb_atomic_spec_from_atomic_numbers(z, native)
        return _pairwise_atomic_spec_from_prepared(prepared)

    def to_dict(self):
        result = {"kind": "dftb"}
        if self.input is not None:
            result["input"] = self.input.to_dict()
        if self.native:
            result["native"] = self.native
        return result

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
                      cell, pos, verbose=True, type_symbols=None):
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
            type_symbols (list of str, optional): Type-index to species-symbol mapping.
                Provide this when the instance will later be used with SKTB setup.
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

        resolved_type_symbols = [] if type_symbols is None else list(type_symbols)
        cpp_obj = _rsatb.RSATB(npar, n_atom, N_atom, atom_offset, n_edge, N_edge,
                               atom_index, atom_type, edge_index, type_norb, resolved_type_symbols,
                               edge_shift_vec, cell, pos, verbose)
        return cls(cpp_obj)

    def setup_model(self, model, options=None):
        """
        Setup the RSATB calculation from typed model/options objects.

        Args:
            model: One of `LCAOModel`, `SKTBModel`, or `DFTBModel`.
            options (SetupOptions, optional): Shared setup options.
        """
        if options is None:
            options = SetupOptions()
        self._cpp_obj.setup_model(model.to_dict(), options.to_dict())

    def setup_lcao(self, model, options=None):
        if options is None:
            options = SetupOptions()
        self._cpp_obj.setup_lcao(model.to_dict(), options.to_dict())

    def setup_sktb(self, model, options=None):
        """
        Setup the RSATB calculation using a typed SKTB model helper.
        """
        if options is None:
            options = SetupOptions()
        self._cpp_obj.setup_sktb(model.to_dict(), options.to_dict())

    def setup_dftb(self, model, options=None):
        if options is None:
            options = SetupOptions()
        self._cpp_obj.setup_dftb(model.to_dict(), options.to_dict())

    def setup_config(self, config):
        """
        Structured setup entry point for operator/model options.

        Expected top-level keys include:
            model, kpoints, nsamples, compute_bounds, preconditioner
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

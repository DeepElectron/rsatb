import _rsatb
import numpy as np

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
        """
        ac_out, omega_out = self._cpp_obj.compute_ac(nt, n_kpm, temp, mu, axis)
        ac_out = np.array(ac_out).reshape(-1, nt+1, 6)
        ntp1 = ac_out.shape[1] # [n_subk, (nt+1)*6]
        
        sigma = np.zeros((-1, 2*ntp1, 6))
        sigma[:, ntp1:, :] = ac_out
        sigma[:, 1:ntp1+1, :] = ac_out[:, ::-1, :]
        ac_out = ac_out + 1j * np.imag(sp.signal.hilbert(sigma, axis=1))[:,ntp1:2*ntp1,:]
        ac_out = ac_out.reshape(-1, (nt+1)*6)

        return ac_out, np.array(omega_out)

    def get_kernel_info(self):
        """
        Get information about the computational kernel.

        Returns:
            str: Kernel information.
        """
        return self._cpp_obj.get_kernel_info()

    def get_operator_construction(self):
        """
        Get the operator construction object.
        """
        return self._cpp_obj.get_operator_construction()

    def atomic_data(self):
        """
        Get the atomic data object.
        """
        return self._cpp_obj.atomic_data()

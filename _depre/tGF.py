from ..core.rsatb import tdos
from ..rsatb.data import AtomicData, _keys, OrbitalMapper, block_to_feature
import numpy as np
import h5py
from ..rsatb.hs.r2k_sparse import R2KSparse
from ..rsatb.linalg.mkl_operator import MKLQuantumOperator
import scipy.sparse as sp
import mpi4py
import pyamg
import ase.io as io

class timeGF:
    def __init__(self, h_indptr, h_indices, h_data, nsamples, n, s_indptr=None, s_indices=None, s_data=None):
        self.h_indptr = h_indptr
        self.h_indices = h_indices
        self.h_data = h_data
        self.s_indptr = s_indptr
        self.s_indices = s_indices
        self.s_data = s_data
        self.nsamples = nsamples
        self.n = n
        self._rescale = 1.
        self._shift = 0.
        
    def get_samples(self):

        if not hasattr(self, "samples"):
            self.samples = np.random.randn(self.n, self.nsamples)
            self.samples /= np.linalg.norm(self.samples, axis=0, keepdims=True)
            self.samples = np.asfortranarray(self.samples)

        return self.samples
    
    def rescale(self, shift, factor):

        self._rescale *= factor
        self._shift += shift
        self.h_data /= factor

    @property
    def H(self):
        return MKLQuantumOperator(self.h_data, self.h_indices, self.h_indptr, shape=(self.n,self.n), upper=True, unit=False)
    
    @property
    def S(self):
        return MKLQuantumOperator(self.s_data, self.s_indices, self.s_indptr, shape=(self.n,self.n), upper=True, unit=True)

    def rescale_auto(self):
        # This function have omp threads (used in amg) overlapping with mkl thread (used by H). Current fix is just use CSR H
        # TODO: write this into c++ code with simple AMG preconditioned BCG with HPDDM, using lib such as AMGCL.
        S = self.S.to_csr()
        H = self.H.to_csr()
        S_csr = sp.csr_matrix((self.s_data, self.s_indices, self.s_indptr), shape=(self.n,self.n), copy=False)
        H_csr = sp.csr_matrix((self.h_data, self.h_indices, self.h_indptr), shape=(self.n,self.n), copy=False)

        M = pyamg.aggregation.smoothed_aggregation_solver(S) # one copy of S

        shape = M.levels[0].A.shape
        dtype = M.levels[0].A.dtype
        def matvec(b):
            return M.solve(b, cycle="V", tol=1e-12)

        Minv = sp.linalg.LinearOperator(shape, matvec, dtype=dtype)

        emax = sp.linalg.eigsh(H, 1, M=S, Minv=Minv, which="LA", tol=1e-3, return_eigenvectors=False)[0]
        emin = sp.linalg.eigsh(H, 1, M=S, Minv=Minv, which="SA", tol=1e-3, return_eigenvectors=False)[0]
        
        del M

        shift = emax + emin
        shift /= 2.

        rescale = emax - emin + 4
        rescale /= 2.

        H_csr -= S_csr * shift # this op will create a new buffer
        H_csr /= rescale

        self.h_indices = H_csr.indices
        self.h_data = H_csr.data
        self.h_indptr = H_csr.indptr

        self._shift += shift
        self._rescale *= rescale

    def get_dt_fft(self):
        return np.pi / self._rescale
    
    def dos(self, nt=512):
        dt = self.get_dt_fft()
        ts = np.cumsum([0] + [dt] * nt)
        ts = ts * self._rescale

        samples = self.get_samples()
        print("Start calculating cdos:")
        corr_dos = tdos(
            self.h_indptr, self.h_indices, self.h_data, samples,
            self.n, self.nsamples, ts,
            self.s_indptr, self.s_indices, self.s_data
        )

        print("End calculating cdos:")

        corr_dos /= np.exp(1j*self._shift*ts/self._rescale)

        assert len(corr_dos) == nt+1

        energies = np.array([i * self._rescale / nt - self._rescale
                             for i in range(nt * 2)])
        
        cdos = np.zeros(nt * 2, dtype=np.complex128)
        dos = np.zeros(nt * 2, dtype=np.double)
        window = 0.5 * (1 + np.cos(np.pi * np.arange(nt).astype(np.complex128) / nt))

        cdos[nt-1:] = corr_dos
        cdos[:nt-1] = corr_dos[::-1][1:-1].conj()

        cdos[nt:] *= window
        cdos[:nt-1] *= window[:-1][::-1]

        cdos = np.fft.ifft(cdos)
        dos[:nt] = np.abs(cdos[nt:2*nt])
        dos[nt:2*nt] = np.abs(cdos[:nt])

        dos = dos / (np.sum(dos) * (self._rescale / nt))

        return energies, dos
    
    def band(self, nt):
        pass

    @staticmethod
    def from_atomicdata(cls):
        pass

    @staticmethod
    def from_file(cls, stru_path, kpoint, basis, r_max, h_path, nsamples, n, s_path=None):
        # read the hamiltonian/overlap from the hdf5 file
        assert len(kpoint.shape) == 1, "It should pass a single kpoint here!"
        atoms = io.read(stru_path)
        H = h5py.File(h_path, "r")
        if s_path is not None:
            S = h5py.File(s_path, "r")
        else:
            S = None
        
        idp = OrbitalMapper(basis=basis)
        atomicdata = AtomicData.from_ase(atoms=atoms, r_max=r_max)
        if S is not None:
            block_to_feature(data=atomicdata, idp=idp, blocks=H["0"], overlap_blocks=S["0"])
            S.close()
        else:
            block_to_feature(data=atomicdata, idp=idp, blocks=H["0"], overlap_blocks=S)

        H.close()

        h2k_sparse = R2KSparse(idp=idp, spin_deg=idp.spin_deg)
        s2k_sparse = R2KSparse(
            idp=idp, 
            spin_deg=idp.spin_deg, 
            edge_field=_keys.EDGE_OVERLAP_KEY, 
            node_field=_keys.NODE_OVERLAP_KEY, 
            out_field=_keys.OVERLAP_KEY
            )
        
        atomicdata = h2k_sparse(atomicdata)
        H = atomicdata[_keys.HAMILTONIAN_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)
        if S is not None:
            atomicdata = s2k_sparse(atomicdata)
            S = atomicdata[_keys.OVERLAP_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)

            return cls(
                h_indptr=H.indptr, h_indices=H.indices, h_data=H.data,
                nsamples=nsamples, n = n,
                s_indptr=S.indptr, s_indices=S.indices, s_data=S.data
            )
        else:
            return cls(
                h_indptr=H.indptr, h_indices=H.indices, h_data=H.data,
                nsamples=nsamples, n = n
            )
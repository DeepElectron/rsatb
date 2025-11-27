from .data import AtomicData, _keys, OrbitalMapper, block_to_feature
import numpy as np
from .utils.parrallel import MPIMultiLevel
import ase.io as io
import os, sys
sys.setdlopenflags(os.RTLD_LAZY)
from rsatb.core import Solver

class RSATB(MPIMultiLevel):
    """
        Wrapper class for c++ rsatb, it handles the MPI parralization of level-1 (kpoints/Energy grid etc...), level-2 MPI is build 
        implicitly in the c++ implementation.
    """
    def __init__(
            self, 
            basis,
            atomicdata: AtomicData,
            n_group:int=1, # maximum number of k in parrallel
            enable_mpi=False,
            verbose=False
            ):
        
        super(RSATB, self).__init__(enable_mpi=enable_mpi, verbose=verbose)
        
        self.idp = OrbitalMapper(basis=basis)
        atomicdata = self.idp(atomicdata) # map the datatype
        self.basis = self.idp.basis
        
        # handling mpi group partition
        self.n_group = n_group
        assert self.n_group > 0, "The number of groups should be a positive number, default by 1."
        if self.n_group > 1:
            self.partition(self.n_group)

        # partition the atomic graph if needed.
        if (not hasattr(atomicdata, _keys.CLUSTER_CONNECTIVITY)) and (self.subsize > 1):
            atomicdata.partition_graph(self.subsize, comm=self.subcomm, split=True)
            atom_index = atomicdata[_keys.ATOM_INDEX_KEY].ravel()
        else:
            atom_index = np.arange(atomicdata.num_nodes)
        # atomicdata.partition(self.n_group)
        
        self.solver = Solver(
            atom_index,
            atomicdata[_keys.ATOM_TYPE_KEY].ravel(), 
            atomicdata[_keys.CELL_KEY],
            atomicdata[_keys.POSITIONS_KEY],
            atomicdata[_keys.EDGE_INDEX_KEY].T,
            self.idp.atom_norb,
            atomicdata[_keys.EDGE_CELL_SHIFT_KEY],
            self.subcomm,
            "-info stacking.log:ksp,pc,eps -log_view :profiling.log"
        ) # c++ implementation
        
        self.__kpoint_set__ = False
        self.__sample_set__ = False

    def setup(self, n, rsmethod, kpoints: np.ndarray, HFile, SFile=None, orblist=None, orbroot=None, tol=1e-7, sptol=1e-6, maxit=20):
        self.setup_sample(n, rsmethod)
        self.setup_kpoints(kpoints)

        self.SFile = SFile
        self.HFile = HFile
        self.orblist = orblist
        self.orbroot = orbroot

        self.tol = tol
        self.sptol = sptol
        self.maxit = maxit
        
        return True

    def setup_kpoints(self, kpoints: np.ndarray):
        """
            1. do the MPI partition according to the kpoint;
            2. call kpoint setup function in the c++ solver.
        """
        if len(kpoints.shape) == 1:
            kpoints = kpoints.reshape(1,-1)

        nk = kpoints.shape[0]
        
        assert len(kpoints.shape) == 2, "kpoint should have shape [nkp, 3]"
        if nk < self.n_group:
            raise RuntimeError("Number of Kpoints is less than the Parrallel groups numbers, Bad Allocation !")

        self.__sub_nk__ = int(np.ceil(float(nk) / self.n_group))
        self.__sub_kpoints__ = kpoints[self.color*self.__sub_nk__:(self.color+1)*self.__sub_nk__] # remark, the number of kpoint in sub_kpoints can be smaller than sub_nk

        self.__kpoint_set__ = True
        self.__kpoint_count__ = 0

        return True
    
    def setup_S_h5(self, SFile, tol=1e-7, sptol=1e-6, maxit=20):
        self.solver.setup_K(*self.__sub_kpoints__[self.__kpoint_count__])
        self.solver.setup_S_h5(SFile)
        self.solver.Ssqrtm(tol, sptol, maxit, self.verbose)

        return True
    
    def setup_S_2c(self, orblist, orbroot, tol=1e-7, sptol=1e-6, maxit=20):
        self.solver.setup_K(*self.__sub_kpoints__[self.__kpoint_count__])
        self.solver.setup_S_orb(orblist, orbroot)
        self.solver.Ssqrtm(tol, sptol, maxit, self.verbose)

        return True
    
    def setup_H_h5(self, HFile, emin=0.0, emax=0.0):
        self.solver.setup_K(*self.__sub_kpoints__[self.__kpoint_count__])
        self.solver.setup_H_h5(HFile, emin, emax, self.verbose)

        return True
    
    def setup_sample(self, n: int=1, method: str="gauss"):
        self.solver.setup_sample(n, method)

        return True
    
    def compute_dos(self, dE:float=0.025852, proj:str="none", orbtype: np.ndarray=np.array([])):

        dos = []
        pdos = []
        omegas = []

        for _ in range(self.__sub_kpoints__.shape[0]):
            if self.orblist is not None and self.orbroot is not None:
                self.setup_S_2c(self.orblist, self.orbroot, self.tol, self.sptol, self.maxit)
            elif self.SFile is not None:
                self.setup_S_h5(self.SFile, self.tol, self.sptol, self.maxit)
            else:
                raise ValueError("The overlap information is not provided in the setup, currently, we only support non-orthogonal basis calculation!")
            
            if self.HFile is not None:
                self.setup_H_h5(self.HFile, 0, 0)
            else:
                raise ValueError("We currently support reading Hamiltonian from H5 file, more will be added later!")
            # compute dos
            nt = self.solver.get_nt(dE)
            out = self.solver.compute_dos(nt, proj, orbtype, self.verbose)
            dos.append(out[0]), pdos.append(out[1]), omegas.append(out[2])
        
        dos = np.asarray(dos)
        pdos = np.asarray(pdos)
        omegas = np.asarray(omegas)

        # MPI summit all rank's dos to the main rank
        # 1. handel the dimension mismatch, this can occur in the last dimension
        diff = 0
        if self.color == (self.n_group-1) and self.n_group > 1:
            if self.group_count[-1] < self.group_count[0]:
                diff = self.group_count[0] - self.group_count[-1]
                dos = np.append(dos, np.zeros([diff]+dos.shape[1:], dtype=dos.dtype), axis=0)
                omegas = np.append(omegas, np.zeros([diff]+omegas.shape[1:], dtype=omegas.dtype), axis=0)

                if proj != "none":
                    pdos = np.append(pdos, np.zeros([diff]+dos.shape[1:], dtype=dos.dtype), axis=0)
        
        # 2. gather the variables, and remove paddings
        dos = self.gather(dos, 0)
        omegas = self.gather(omegas, 0)

        if self.rank == 0:
            if diff > 0:
                dos = dos[:-diff]
            if proj != "none":
                pdos = self.gather(pdos, 0)
                if diff > 0:
                    pdos = pdos[:-diff]

            if proj != "none":
                return omegas, dos, pdos
            else:
                return omegas, dos
        else:
            return True
    
    # def dos(self, nt=512):
    #     dt = self.get_dt_fft()
    #     ts = np.cumsum([0] + [dt] * nt)
    #     ts = ts * self._rescale

    #     samples = self.get_samples()
    #     print("Start calculating cdos:")
    #     corr_dos = tdos(
    #         self.h_indptr, self.h_indices, self.h_data, samples,
    #         self.n, self.nsamples, ts,
    #         self.s_indptr, self.s_indices, self.s_data
    #     )

    #     print("End calculating cdos:")

    #     corr_dos /= np.exp(1j*self._shift*ts/self._rescale)

    #     assert len(corr_dos) == nt+1

    #     energies = np.array([i * self._rescale / nt - self._rescale
    #                          for i in range(nt * 2)])
        
    #     cdos = np.zeros(nt * 2, dtype=np.complex128)
    #     dos = np.zeros(nt * 2, dtype=np.double)
    #     window = 0.5 * (1 + np.cos(np.pi * np.arange(nt).astype(np.complex128) / nt))

    #     cdos[nt-1:] = corr_dos
    #     cdos[:nt-1] = corr_dos[::-1][1:-1].conj()

    #     cdos[nt:] *= window
    #     cdos[:nt-1] *= window[:-1][::-1]

    #     cdos = np.fft.ifft(cdos)
    #     dos[:nt] = np.abs(cdos[nt:2*nt])
    #     dos[nt:2*nt] = np.abs(cdos[:nt])

    #     dos = dos / (np.sum(dos) * (self._rescale / nt))

    #     return energies, dos
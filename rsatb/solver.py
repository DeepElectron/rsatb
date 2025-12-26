from .data import AtomicData, _keys, OrbitalMapper, block_to_feature
import numpy as np
from .utils.parrallel import MPIMultiLevel
import ase.io as io
import os, sys
import scipy as sp
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
            atomicdata.partition_graph_hilbert(self.subsize, comm=self.subcomm, split=True)
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

    def setup(self, n=1, rsmethod="gauss", kpoints=np.array([[0.,0.,0.]]), HFile=None, SFile=None, RxFile=None, RyFile=None, RzFile=None, orblist=None, orbroot=None, tol=1e-7, sptol=1e-6, maxit=20, orthogonalize=False, explicit_orth=True):
        self.setup_sample(n, rsmethod)
        self.setup_kpoints(kpoints)

        self.SFile = SFile
        self.HFile = HFile
        self.RxFile = RxFile
        self.RyFile = RyFile
        self.RzFile = RzFile
        self.orblist = orblist
        self.orbroot = orbroot
        self.orthogonalize = orthogonalize
        self.explicit_orth = explicit_orth

        if not self.orthogonalize and self.explicit_orth:
            raise RuntimeWarning("explicit_orth have no effect when you set orthogonalize = False!")

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
    
    def _setup_kpoint(self, kidx, need_R=False):
        self.solver.setup_K(*self.__sub_kpoints__[kidx])
        
        # Setup S and R
        # Optimization: If we need R and are using orbitals, setup_R_orb can setup both S and R at once.
        if need_R and (self.orblist is not None and self.orbroot is not None):
            self.solver.setup_R_orb(self.orblist, self.orbroot)
            # S is now setup by setup_R_orb if it wasn't already
        elif self.orblist is not None and self.orbroot is not None:
            self.solver.setup_S_orb(self.orblist, self.orbroot)
        elif self.SFile is not None:
            self.solver.setup_S_h5(self.SFile)
        else:
            raise ValueError("The overlap information is not provided in the setup, currently, we only support non-orthogonal basis calculation!")
        
        if self.HFile is not None:
            self.solver.setup_H_h5(self.HFile)
        else:
            raise ValueError("We currently support reading Hamiltonian from H5 file, more will be added later!")

        if self.orthogonalize:
            self.solver.orthogonalize_H(self.tol, self.sptol, self.maxit, self.explicit_orth, self.verbose)
        else:
            self.solver.Ssqrtm(self.tol, self.sptol, self.maxit, self.verbose, False)
            self.solver.setup_KSP()
        
        self.solver.scale_shift_H(0.0, 0.0, self.verbose)
        
        # Setup R (if needed and not done yet)
        if need_R:
            # If we used orbitals, it's already done above.
            # If we used H5, we need to setup R from files.
            if self.orblist is None: # Assuming H5 path
                 self.solver.setup_R_h5(self.RxFile if self.RxFile else "", 
                                        self.RyFile if self.RyFile else "", 
                                        self.RzFile if self.RzFile else "")

    def setup_sample(self, n: int=1, method: str="gauss"):
        self.solver.setup_sample(n, method)

        return True
    
    def compute(self, tasks: list):
        """
        Execute a list of tasks (DOS, AC) efficiently.
        tasks: list of dicts, e.g. [{'task': 'dos', 'dE': ...}, {'task': 'ac', ...}]
        """
        results = [[] for _ in tasks]
        
        # Check if any task needs R
        need_R = any(t['task'] == 'ac' for t in tasks)
        if need_R and np.abs(self.__sub_kpoints__).sum() > 0:
            raise ValueError("AC computation only support Gamma Point!")

        for i in range(self.__sub_kpoints__.shape[0]):
            self._setup_kpoint(i, need_R=need_R)
            
            for t_idx, task in enumerate(tasks):
                if task['task'] == 'dos':
                    dE = task.get('dE', 0.025852)
                    proj = task.get('proj', "none")
                    orbtype = task.get('orbtype', np.array([]))
                    
                    nt = self.solver.get_nt(dE)
                    # Note: compute_dos in C++ handles its own logic, we just call it.
                    # It uses the currently setup matrices.
                    
                    # We need to handle the output pointers.
                    # The C++ binding returns a tuple (dos, pdos, omegas)
                    out = self.solver.compute_dos(nt, proj, orbtype, self.verbose)
                    results[t_idx].append(out)
                    
                elif task['task'] == 'ac':
                    dE = task.get('dE', 0.025852)
                    n_kpm = task.get('n_kpm', 200)
                    temp = task.get('temp', 300)
                    mu = task.get('mu', 0.0)
                    
                    nt = task.get('nt', None)
                    if nt is None:
                        nt = self.solver.get_nt(dE) # This returns void in C++, need to check binding
                        # Actually get_nt in C++ returns void and takes int* pointer.
                        # Easier to use get_scale() and calc in Python.
                    
                    out = self.solver.compute_ac(nt, n_kpm, temp, mu)
                    results[t_idx].append(out)

        # Gather results
        final_results = []
        
        # Determine padding for MPI gather
        diff = 0
        if self.color == (self.n_group-1) and self.n_group > 1:
            if self.group_count[-1] < self.group_count[0]:
                diff = self.group_count[0] - self.group_count[-1]

        for t_idx, task in enumerate(tasks):
            task_res = results[t_idx] # List of tuples from each kpoint
            
            if task['task'] == 'dos':
                # Unpack list of tuples [(dos, pdos, omegas), ...]
                # Structure: dos [2*nt], pdos [2*nt*nproj], omegas [2*nt]
                # We want to stack them: [nk, 2*nt]
                
                dos_list = [r[0] for r in task_res]
                pdos_list = [r[1] for r in task_res]
                omegas_list = [r[2] for r in task_res]
                
                dos_arr = np.asarray(dos_list)
                pdos_arr = np.asarray(pdos_list)
                omegas_arr = np.asarray(omegas_list)
                
                # Padding
                if diff > 0:
                    dos_arr = np.append(dos_arr, np.zeros([diff]+list(dos_arr.shape[1:]), dtype=dos_arr.dtype), axis=0)
                    omegas_arr = np.append(omegas_arr, np.zeros([diff]+list(omegas_arr.shape[1:]), dtype=omegas_arr.dtype), axis=0)
                    if task.get('proj', 'none') != 'none':
                        pdos_arr = np.append(pdos_arr, np.zeros([diff]+list(pdos_arr.shape[1:]), dtype=pdos_arr.dtype), axis=0)

                # Gather
                dos_arr = self.gather(dos_arr, 0)
                omegas_arr = self.gather(omegas_arr, 0)
                if task.get('proj', 'none') != 'none':
                    pdos_arr = self.gather(pdos_arr, 0)

                if self.rank == 0:
                    if diff > 0:
                        dos_arr = dos_arr[:-diff]
                        omegas_arr = omegas_arr[:-diff]
                        if task.get('proj', 'none') != 'none':
                            pdos_arr = pdos_arr[:-diff]
                    
                    if task.get('proj', 'none') != 'none':
                        final_results.append((omegas_arr, dos_arr, pdos_arr))
                    else:
                        final_results.append((omegas_arr, dos_arr))
                else:
                    final_results.append(True)

            elif task['task'] == 'ac':
                # Unpack [(ac, omegas), ...]
                ac_list = [r[0] for r in task_res]
                omegas_list = [r[1] for r in task_res]
                
                ac_arr = np.asarray(ac_list)
                omegas_arr = np.asarray(omegas_list)
                
                if diff > 0:
                    ac_arr = np.append(ac_arr, np.zeros([diff]+list(ac_arr.shape[1:]), dtype=ac_arr.dtype), axis=0)
                    omegas_arr = np.append(omegas_arr, np.zeros([diff]+list(omegas_arr.shape[1:]), dtype=omegas_arr.dtype), axis=0)
                
                if self.is_submaster:
                    # transform ac
                    nsubk = ac_arr.shape[0]
                    ac_arr = ac_arr.reshape(nsubk, -1, 6)
                    ntp1 = ac_arr.shape[1] # [n_subk, (nt+1)*6]
                    
                    
                    sigma = np.zeros((nsubk, 2*ntp1, 6))
                    sigma[:, ntp1:, :] = ac_arr
                    sigma[:, 1:ntp1+1, :] = ac_arr[:, ::-1, :]
                    ac_arr = ac_arr + 1j * np.imag(sp.signal.hilbert(sigma, axis=1))[:,ntp1:2*ntp1,:]

                    ac_arr = ac_arr.reshape(nsubk, -1)

                ac_arr = self.gather(ac_arr, 0)
                omegas_arr = self.gather(omegas_arr, 0)
                
                if self.rank == 0:
                    if diff > 0:
                        ac_arr = ac_arr[:-diff]
                        omegas_arr = omegas_arr[:-diff]
                    final_results.append((omegas_arr, ac_arr))
                else:
                    final_results.append(True)
                    
        return final_results

    def compute_dos(self, dE:float=0.025852, proj:str="none", orbtype: np.ndarray=np.array([])):
        task = {'task': 'dos', 'dE': dE, 'proj': proj, 'orbtype': orbtype}
        return self.compute([task])[0]

    def compute_ac(self, dE:float=0.025852, nt:int=None, n_kpm:int=200, temp:float=300, mu:float=0.0):
        task = {'task': 'ac', 'dE': dE, 'nt': nt, 'n_kpm': n_kpm, 'temp': temp, 'mu': mu}
        return self.compute([task])[0]
    
    def export_H(self, root):
        self.solver.export_H(root)
        return True
    
    def export_S(self, root):
        self.solver.export_S(root)
        return True
    
    def export_Rx(self, root):
        self.solver.export_Rx(root)
        return True
    
    def export_Ry(self, root):
        self.solver.export_Ry(root)
        return True
    
    def export_Rz(self, root):
        self.solver.export_Rz(root)
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
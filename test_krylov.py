from src.data import AtomicData, _keys, OrbitalMapper, block_to_feature
from src.hs.r2k import R2K
from src.hs.r2k_sparse import R2KSparse
from src.linalg.krylov import krylov_arnoldi, bkrylov_arnoldi
from ase.io import read
import numpy as np
import scipy as sp
import h5py
import pyamg


sysname = "Al"
atomicdata = AtomicData.from_ase(read("./src/test/data/"+sysname+".vasp"), r_max=7.4)
atomicdata = atomicdata.to_dict()
atomicdata[_keys.KPOINT_KEY] = np.array([[0.,0.,0.]])
idp = OrbitalMapper(basis={"Al": "4s4p1d"})

h2k = R2K(idp=idp, spin_deg=idp.spin_deg)
s2k = R2K(
    idp=idp, 
    spin_deg=idp.spin_deg, 
    edge_field=_keys.EDGE_OVERLAP_KEY, 
    node_field=_keys.NODE_OVERLAP_KEY, 
    out_field=_keys.OVERLAP_KEY
    )
h2k_sparse = R2KSparse(idp=idp, spin_deg=idp.spin_deg)
s2k_sparse = R2KSparse(
    idp=idp, 
    spin_deg=idp.spin_deg, 
    edge_field=_keys.EDGE_OVERLAP_KEY, 
    node_field=_keys.NODE_OVERLAP_KEY, 
    out_field=_keys.OVERLAP_KEY
    )

H = h5py.File("./src/test/data/H_"+sysname+".h5", "r")["0"]
S = h5py.File("./src/test/data/S_"+sysname+".h5", "r")["0"]
block_to_feature(data=atomicdata, idp=idp, blocks=H, overlap_blocks=S)

atomicdata = h2k_sparse(atomicdata)
atomicdata = s2k_sparse(atomicdata)

def test_krylov():
    H_ = atomicdata[_keys.HAMILTONIAN_KEY][0].to_scipy_sparse().tocsr().astype(np.float64)
    S_ = atomicdata[_keys.OVERLAP_KEY][0].to_scipy_sparse().tocsr().astype(np.float64)
    M = pyamg.aggregation.smoothed_aggregation_solver(S_)
    M = M.aspreconditioner()
    Q, _, _ = krylov_arnoldi(
        a=H_, 
        b=S_, 
        M=M,
        implicit=True,
        reorth=False,
        v=100,
        )
    H = Q.conj().T @ H_ @ Q
    S = Q.conj().T @ S_ @ Q
    print(np.abs(S-np.eye(S.shape[0])).max())

    import matplotlib.pyplot as plt
    plt.matshow(H.real, cmap="bwr", vmax=1., vmin=-1.)
    plt.savefig("./H.png", dpi=100)
    plt.close()

    eigs, vecs = sp.linalg.eig(a=H)
    eigs_sparse, vecs_sparse = sp.linalg.eigh(atomicdata[_keys.HAMILTONIAN_KEY][0].todense(), atomicdata[_keys.OVERLAP_KEY][0].todense())

    print(np.abs(eigs[:3] - eigs_sparse[:3]))
    assert np.abs(eigs[:3] - eigs_sparse[:3]).max() < 1e-3

def test_bkrylov():
    import time
    H_ = atomicdata[_keys.HAMILTONIAN_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)
    S_ = atomicdata[_keys.OVERLAP_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)
    M = pyamg.aggregation.smoothed_aggregation_solver(S_)
    M = M.aspreconditioner()
    v0 = np.random.randn(1,H_.shape[0])

    start = time.time()
    Q, _, _ = bkrylov_arnoldi(
        a=H_, 
        b=S_,
        v0=v0,
        implicit=True,
        reorth=True,
        v=10,
        )
    
    print(time.time()-start, "bkrylov time")
    start = time.time()
    Q_lbl, _, _ = krylov_arnoldi(
        a=H_, 
        b=S_,
        M=M,
        v0=v0[0],
        implicit=True,
        reorth=True,
        v=10,
        )
    
    print(time.time()-start, "krylov time")
    
    assert np.abs(Q[0] - Q_lbl).max() < 1e-6, print(Q[0] - Q_lbl)

if __name__ == "__main__":
    test_krylov()
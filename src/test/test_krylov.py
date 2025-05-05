from src.data import AtomicData, _keys, OrbitalMapper, block_to_feature
from src.hs.r2k import R2K
from src.hs.r2k_sparse import R2KSparse
from src.linalg.krylov import Krylov_Arnoldi
from ase.io import read
import numpy as np
import scipy as sp
import h5py


sysname = "Al555"
atomicdata = AtomicData.from_ase(read("./data/"+sysname+".vasp"), r_max=7.4)
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

H = h5py.File("./data/H_"+sysname+".h5", "r")["0"]
S = h5py.File("./data/S_"+sysname+".h5", "r")["0"]
block_to_feature(data=atomicdata, idp=idp, blocks=H, overlap_blocks=S)

atomicdata = h2k_sparse(atomicdata)
atomicdata = s2k_sparse(atomicdata)

def test_krylov():
    Q = Krylov_Arnoldi(a=atomicdata[_keys.HAMILTONIAN_KEY][0].to_scipy_sparse(), b=atomicdata[_keys.OVERLAP_KEY][0].to_scipy_sparse())
    H = Q.conj().T @ atomicdata[_keys.HAMILTONIAN_KEY][0] @ Q
    S = Q.conj().T @ atomicdata[_keys.OVERLAP_KEY][0] @ Q

    eigs, vecs = sp.linalg.eigh(a=H, b=S)
    eigs_sparse, vecs_sparse = sp.linalg.eigh(atomicdata[_keys.HAMILTONIAN_KEY][0].todense(), atomicdata[_keys.OVERLAP_KEY][0].todense())

    assert np.abs(eigs[:10] - eigs_sparse[:10]).max() < 1e-3

if __name__ == "__main__":
    test_krylov()
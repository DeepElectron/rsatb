from src.data import AtomicData, _keys, OrbitalMapper, block_to_feature
from src.hs.r2k import R2K
from src.hs.r2k_sparse import R2KSparse
from ase.io import read
import numpy as np
import scipy as sp
import h5py
from src.solver.tGF import efermi
# np.random.seed(1234)

sysname = "Al"
atomicdata = AtomicData.from_ase(read("./src/test/data/"+sysname+".vasp"), r_max=7.4)
atomicdata = atomicdata.to_dict()
atomicdata[_keys.KPOINT_KEY] = np.array([[0.,0.2,-0.2]])
idp = OrbitalMapper(basis={"Al": "4s4p1d"})

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

def test_efermi():
    S = 1000
    H_ = atomicdata[_keys.HAMILTONIAN_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)
    S_ = atomicdata[_keys.OVERLAP_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)

    # vals, _ = sp.linalg.eigh(H_.todense(), S_.todense())

    v0 = np.random.randn(H_.shape[0], S).astype(H_.dtype)
    ef = efermi(
        A_csr=H_, 
        B_csr=S_, 
        v0=v0,
        nocc=4,
        verbose=True,
        Mcut=1000,
        R=1e5,
        kBT=0.025,
        ntol=1e-3,
    )

    print(ef)




if __name__ == "__main__":
    test_efermi()

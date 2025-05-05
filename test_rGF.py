from src.data import AtomicData, _keys, OrbitalMapper, block_to_feature
from src.hs.r2k import R2K
from src.hs.r2k_sparse import R2KSparse
from src.solver.rGF import random_greens_function
from ase.io import read
import numpy as np
import scipy as sp
import h5py

# np.random.seed(1234)

sysname = "Al333"
atomicdata = AtomicData.from_ase(read("./src/test/data/"+sysname+".vasp"), r_max=7.4)
atomicdata = atomicdata.to_dict()
atomicdata[_keys.KPOINT_KEY] = np.array([[0.,-0.2,0.2]])
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

def test_rGF():
    H_ = atomicdata[_keys.HAMILTONIAN_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)
    S_ = atomicdata[_keys.OVERLAP_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)
    Krd = int(2.2*(H_.shape[0]**0.5))
    print("Kdim: ", Krd)

    # v0 = np.random.randn(1, H_.shape[0]).astype(H_.dtype)

    # print(sp.sparse.linalg.eigsh(H_, M=S_, k=10)[0])
    # for k in range(1,100):
    print(np.max(H_.data.imag))
    v0Gv0, v0 = random_greens_function(
        H=H_,
        S=S_,
        # v0=v0,
        Es=np.array([16.]),
        Nsamples=10000,
        Krmin=180
    )

    # some question: Krmin is too large!
    # direct solver is not fast enough, not good for testing.

    # v0Gv0_ = ((sp.sparse.linalg.spsolve((16.+1e-4*1j)*S_-H_, (S_ @ v0.T[:,0]))[None,:]) * v0.conj()).sum(axis=-1) # work when nsample = 1
    TrG = np.linalg.trace((sp.linalg.inv((16.+1e-4*1j)*S_-H_.todense()) @ S_.todense()))

    # this take some time to run
    # print(v0Gv0, v0Gv0_)
    # print(np.abs(v0Gv0_-v0Gv0)[0])

    print(TrG, v0Gv0)

if __name__ == "__main__":
    test_rGF()
import scipy as sp
import mpi4py
from src.data import AtomicData, _keys, OrbitalMapper, block_to_feature
from ase.io import read
import numpy as np
import h5py
from src.hs.r2k_sparse import R2KSparse
import ctypes

sysname = "Al"
atomicdata = AtomicData.from_ase(read("./src/test/data/"+sysname+".vasp"), r_max=7.4)
atomicdata = atomicdata.to_dict()
atomicdata[_keys.KPOINT_KEY] = np.array([[0.,-0.2,0.3]])
idp = OrbitalMapper(basis={"Al": "4s4p1d"})

h2k_sparse = R2KSparse(idp=idp, spin_deg=idp.spin_deg, upper=True)
s2k_sparse = R2KSparse(
    idp=idp, 
    spin_deg=idp.spin_deg, 
    edge_field=_keys.EDGE_OVERLAP_KEY, 
    node_field=_keys.NODE_OVERLAP_KEY, 
    out_field=_keys.OVERLAP_KEY,
    upper=True
    )

H = h5py.File("./src/test/data/H_"+sysname+".h5", "r")["0"]
S = h5py.File("./src/test/data/S_"+sysname+".h5", "r")["0"]
block_to_feature(data=atomicdata, idp=idp, blocks=H, overlap_blocks=S)

atomicdata = h2k_sparse(atomicdata)
atomicdata = s2k_sparse(atomicdata)

H_ = atomicdata[_keys.HAMILTONIAN_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)
S_ = atomicdata[_keys.OVERLAP_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)

from src.solver.rsatb import tdos

ih, jh, h = H_.indptr, H_.indices, H_.data
ts = np.cumsum([0]+[np.pi/110]*20)
_is, js, s = S_.indptr, S_.indices, S_.data
Psi_in = np.random.randn(H_.shape[0], 1)
Psi_in = Psi_in / np.linalg.norm(Psi_in, axis=0, keepdims=True)
Psi_in = np.asfortranarray(Psi_in)

H__ = H_.todense()
H__ = H__ + H__.conj().T
H__[np.diag_indices(H_.shape[0])] /= 2.

S__ = S_.todense()
S__ = S__ + S__.conj().T
S__[np.diag_indices(S_.shape[0])] /= 2.
from scipy.linalg import expm
op = np.linalg.inv(S__) @  H__
op = np.stack([expm(-1j * ts[i] * op) for i in range(len(ts))])
label = (np.einsum("tjk,kl->tjl", op, Psi_in) * Psi_in.conj()[None,:,:]).mean(axis=-1).sum(axis=-1) * H__.shape[0]

normh = 110
h = h / normh
ts = ts * normh


if __name__ == "__main__":
    print("Begin tdos compute!")
    result = tdos(
        ih, jh, h, Psi_in,
        H_.shape[0], Psi_in.shape[1], ts,
        _is, js, s
    )

    print(label - result, "abs error")
    print(label, "label")
    print(result, "output")
    print((label - result) / np.abs(label), "rel error")
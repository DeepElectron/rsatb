from rsatb.data import AtomicData, _keys, OrbitalMapper, block_to_feature
import h5py
from rsatb.hs.r2k_sparse import R2KSparse
from ase.io import read
import numpy as np
import mpi4py

sysname = "Al202020"
atomicdata = AtomicData.from_ase(read("./data/"+sysname+".vasp"), r_max=7.4)
atomicdata = atomicdata.to_dict()
atomicdata[_keys.KPOINT_KEY] = np.array([[0.,0.,0.]])
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

H = h5py.File("./data/H_"+sysname+".h5", "r")["0"]
S = h5py.File("./data/S_"+sysname+".h5", "r")["0"]
block_to_feature(data=atomicdata, idp=idp, blocks=H, overlap_blocks=S)

atomicdata = h2k_sparse(atomicdata)
atomicdata = s2k_sparse(atomicdata)

H_ = atomicdata[_keys.HAMILTONIAN_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)
S_ = atomicdata[_keys.OVERLAP_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)

# H__ = H_.todense()
# S__ = S_.todense()

ih, jh, h = H_.indptr, H_.indices, H_.data
_is, js, s = S_.indptr, S_.indices, S_.data

from rsatb.solver.tGF import timeGF
import scipy as sp
import matplotlib.pyplot as plt
tgf = timeGF(
    h_indptr=ih,
    h_indices=jh,
    h_data=h,
    nsamples=1,
    n=H_.shape[0],
    s_indptr=_is,
    s_indices=js,
    s_data=s,
)

tgf.rescale_auto()

energies, outdos = tgf.dos(nt=256)

plt.plot(energies, outdos)
plt.savefig("dos_rescale.png", dpi=120)
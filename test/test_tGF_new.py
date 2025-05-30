from src.data import AtomicData, _keys, OrbitalMapper, block_to_feature
import h5py
from src.hs.r2k_sparse import R2KSparse
from ase.io import read
import numpy as np
import mpi4py

sysname = "Al282828"
atomicdata = AtomicData.from_ase(read("./src/test/data/"+sysname+".vasp"), r_max=7.4)
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

H = h5py.File("./src/test/data/H_"+sysname+".h5", "r")["0"]
S = h5py.File("./src/test/data/S_"+sysname+".h5", "r")["0"]
block_to_feature(data=atomicdata, idp=idp, blocks=H, overlap_blocks=S)

atomicdata = h2k_sparse(atomicdata)
atomicdata = s2k_sparse(atomicdata)

H_ = atomicdata[_keys.HAMILTONIAN_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)
S_ = atomicdata[_keys.OVERLAP_KEY][0].to_scipy_sparse().tocsr().astype(np.complex128)

# H__ = H_.todense()
# S__ = S_.todense()

ih, jh, h = H_.indptr, H_.indices, H_.data
_is, js, s = S_.indptr, S_.indices, S_.data


if __name__ == "__main__":
    from src.solver.tGF import timeGF
    from src.solver.rsatb import tdos
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

    tgf.rescale(110.)

    energies, outdos = tgf.dos(nt=512)

    plt.plot(energies, outdos)
    plt.savefig("dos.png", dpi=120)

    import numpy as np

    np.savez("./dos.npz", e=energies, dos=outdos)

    
    # H__ += H__.conj().T
    # S__ += S__.conj().T
    # H__ -= np.diag(np.diag(H__)) * 0.5
    # S__ -= np.diag(np.diag(S__)) * 0.5

    # print(sp.linalg.eigvalsh(H__, S__))
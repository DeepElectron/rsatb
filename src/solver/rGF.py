# This is the class for random green's function solver
from ..linalg.krylov import blanczos
import numpy as np
import time

def random_greens_function(H, S, Es, Nsamples, v0=None, Krmin=10, Krtol=1e-6, eta=1e-4):
    # generate the Krylov subspace
    # Krmin = 2.2*(N**0.5)
    # first give Krmin dim, then generate a random vector, multiply until converge to a accuracy
    if v0 is None:
        v0 = np.random.randn(Nsamples, H.shape[0]).astype(H.dtype)
    else:
        assert v0.shape == (Nsamples, H.shape[0])

    start = time.time()
    v0Q, alphas, betas = blanczos(A=H, v0=v0, B=S, v=Krmin, reorth=False)
    print(time.time() - start, "krylov time")

    start = time.time()
    # construct the projected Hamiltonian (overlap is I now)
    Hp = np.zeros((Nsamples, Krmin+1, Krmin)).astype(H.dtype)
    ind = np.arange(Krmin)
    Hp[:,ind,ind] = alphas
    Hp[:,ind[1:],ind[:-1]] = betas[:,:-1]
    Hp[:,ind[:-1],ind[1:]] = betas[:,:-1].conj()
    Hp[:,-1, -1] = betas[:,-1]

    # compute the Greens Function
    v0HSv0 = np.sqrt(np.sum((S @ v0.T).T * v0.conj(), axis=-1)) # [bsz]
    assert Es.ndim == 1

    # computation for each E is paralizable
    TrG = []
    for e in Es:
        Gp = np.linalg.pinv((e+eta*1j)*np.eye(Krmin+1,Krmin)[None,...]-Hp) # MINRES
        # Gp = np.linalg.inv((e+eta*1j)*np.eye(Krmin,Krmin)[None,...]-Hp[:,:-1,:]) # CG
        GpQsHSv0 = v0HSv0[:,None] * Gp[:,:,0] # [s,k]
        # Compute full G
        # G = np.einsum("sNk,sk,Ps->NP", Qs, GpQsHSv0, v0.conj().T) / Nsamples # This can be highly optimized
        # Compute diagonal of G
        # Gd = np.einsum("sNk,sk,Ns->N", Qs, GpQsHSv0, v0.conj().T) / Nsamples # This can be highly optimized
        v0Gv0 = (v0Q * GpQsHSv0).sum(axis=-1)
        TrG.append(v0Gv0.mean())
        

    print(time.time() - start, "sampling time")

    return np.stack(TrG), v0
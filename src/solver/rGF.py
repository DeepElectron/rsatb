# This is the class for random green's function solver
from ..linalg.krylov import blanczos
import numpy as np
import time
import scipy.sparse as sp

def _rgf(H, S, Nsamples, v0=None, Krmin=10, Krtol=1e-6, return_Q=False):
    # generate the Krylov subspace
    # Krmin = 2.2*(N**0.5)
    # first give Krmin dim, then generate a random vector, multiply until converge to a accuracy
    if v0 is None:
        v0 = np.random.randn(Nsamples, H.shape[0]).astype(H.dtype)
    else:
        assert v0.shape == (Nsamples, H.shape[0])

    v0Q, alphas, betas, rQ = blanczos(A=H, v0=v0, B=S, v=Krmin, reorth=False, return_Q=return_Q)

    # construct the projected Hamiltonian (overlap is I now)
    Hp = np.zeros((Nsamples, Krmin+1, Krmin)).astype(H.dtype)
    ind = np.arange(Krmin)
    Hp[:,ind,ind] = alphas
    Hp[:,ind[1:],ind[:-1]] = betas[:,:-1]
    Hp[:,ind[:-1],ind[1:]] = betas[:,:-1].conj()
    Hp[:,-1, -1] = betas[:,-1]

    return Hp, v0Q, rQ

def dm_rgf(H, S, mu, Nsamples, v0=None, Krmin=10, Krtol=1e-6, kBT=0.025):
    Hp, v0Q, rQ = _rgf(H, S, Nsamples, v0, Krmin, Krtol, return_Q=True)

    if S is None:
        w0 = v0
    else:
        w0 = sp.linalg.spsolve(S, v0.T).T

    w0norm = np.sqrt(np.sum(w0.conj().T * (S @ w0.T), axis=0))
    Hp, v0Q, rQ = _rgf(H, S, Nsamples, w0, Krmin, 0., return_Q=True)

    valsp, vecsp = np.linalg.eigh(Hp[:,:-1])
    dm = (vecsp * fermi_dirac(valsp, mu, kBT)[:, None, :]) @ vecsp.transpose(0,2,1).conj()
    dm = w0norm[:,None,None] * (rQ @ dm[:,:,[0]]) @ v0.conj()[:,None]
    dm = dm.mean(0)

    # indptr, col_ids = H.indptr, H.indices
    # row_ids = np.repeat(np.arange(0, H.shape[1]), np.diff(indptr))
    # data = (dm[:, row_ids] * v0H[:, col_ids]).mean(axis=0)

    # dm = sp.csr_matrix(
    #     (data, (row_ids, col_ids)), shape=H.shape
    # )
    return dm

    # we assume that the sparsity pattern of dm and H, and S are the same.
    
    
    

def dos_rgf(H, S, Es, Nsamples, v0=None, Krmin=10, Krtol=1e-6, eta=1e-4):
    Hp, v0Q, _ = _rgf(H, S, Nsamples, v0, Krmin, Krtol)

    # compute the trace of the Greens Function
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

    return np.stack(TrG), v0

    

# def random_greens_function(H, S, Es, Nsamples, v0=None, Krmin=10, Krtol=1e-6, eta=1e-4):

def fermi_dirac(energy, E_fermi, kBT):
    x = (energy - E_fermi) / kBT
    if isinstance(x, (int, float)):
        x = np.array(x)
    out = np.zeros_like(x)
    out[x < -500] = 1.
    mask = (x > -500) * (x < 500)
    out[mask] = 1./(np.exp(x[mask]) + 1)
    return out
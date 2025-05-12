import numpy as np
import scipy as sp
import sksparse.cholmod as cholmod
from sparse_dot_mkl import dot_product_mkl
import sparse
import pyamg
import warnings
from .bsolve import bsolve, bsolve_petsc
import time
from petsc4py import PETSc
from .petscop import Tconj, T, rowsum, colsum, dense_to_petsc, csr_to_sbaij, getsub, csr_to_aij

def icgs(u,Q,M=None,return_norm=False,maxiter=30):
    '''
    Iterative Classical M-orthogonal Gram-Schmidt orthogonalization.

    Parameters:
        :u: vector, the column vector to be orthogonalized. [matdim]
        :Q: matrix, the search space. [matdim, Krdim]
        :M: matrix/None, the matrix, if provided, perform M-orthogonal.
        :return_norm: bool, return the norm of u.
        :maxiter: int, the maximum number of iteractions.

    Return:
        vector, orthogonalized vector u.
    '''
    assert(np.ndim(u)==1)
    uH,QH=u.conj(),Q.conj().T
    alpha=0.5
    Mu=M.dot(u) if M is not None else u
    r_pre=uH.dot(Mu)
    for _ in range(maxiter):
        u=u-Q.dot(QH.dot(Mu))
        Mu=M.dot(u) if M is not None else u
        r1=uH.dot(Mu)
        if r1>alpha*r_pre:
            break
        r_pre=r1
    if r1<=alpha*r_pre:
        warnings.warn('loss of orthogonality @icgs.')
    return (u,r1) if return_norm else u

def bicgs(u,Q,M=None,return_norm=False,maxiter=30):
    '''
    Iterative Classical M-orthogonal Gram-Schmidt orthogonalization.

    Parameters:
        :u: vector, the column vector to be orthogonalized. [bsz, matdim]
        :Q: matrix, the search space. [bsz, matdim, Krdim]
        :M: matrix/None, the matrix, if provided, perform M-orthogonal. [matdim, matdim]
        :return_norm: bool, return the norm of u.
        :maxiter: int, the maximum number of iteractions.

    Return:
        vector, orthogonalized vector u.
    '''
    assert np.ndim(u)==2 and u.shape[:2] == Q.shape[:2]
    # put bsz dim in front
    uH,QH=u.conj(),Q.transpose(0,2,1).conj() # [bsz, matdim], [bsz, Krdim, matdim]
    alpha=0.5
    Mu=(M @ u.T).T if M is not None else u # [bsz, matdim]
    r_pre = (uH * Mu).sum(axis=1) # [bsz, matdim] x [bsz, matdim] -> [bsz, M, M] -> [bsz]
    for _ in range(maxiter):
        u=u-(Q @ (QH @ Mu[:,:,None]))[:,:,0] #[bsz, matdim, Krdim], [bsz, Krdim, matdim], [bsz, matdim,1] -> [bsz, matdim, 1] -> [bsz, matdim]
        Mu=(M @ u.T).T if M is not None else u # [bsz, matdim]
        r1=(uH * Mu).sum(axis=1)
        if all(r1>alpha*r_pre):
            break
        r_pre=r1
    if any(r1<=alpha*r_pre):
        warnings.warn('loss of orthogonality @icgs.')
    return (u,r1) if return_norm else u


def lanczos(a, v0=None, b=None, M=None, v=3, implicit=True, reorth=True):
    """
        Arnoldi iteractions to construct Krylov subspace for problems av=kbv.
        The basis can be construct using two mathod, decided by implicit tag.
        For safty, the reorth is required when implicit is True.

        Parameters:
            :a: matrix, both sparse and dense format is supported.
            :v0: the initial basis, should be 1d vector with the same length as :a:.
            :b: the positive definite matrix, defining the inner product of basis of a.
            :M: Preconditioner for compute inverse action of b on v, used when :implicit: is True.
            :v: number of dimension of Krylov subspace to generate.
            :implicit: Deside whether we use:
              True: the standard Krylov_Arnoldi route by treating operator as b^{-1}a. 
                    Then action take tens of SMVP's time, but orthogonalization is simple. (default)
              False: the explicit overlap metric Arnoldi process. The orthogonalization process 
                    need to be taken with all previous genereated basis. The action taken only a|v> or b|v>, 
                    but orthogonalization takes several iteration.
            :reorth: orthogonalize of current vector w.r.t. all previous basis again, increase numerical 
                    stability but have additional cost.
        
        Return:
            Q: the generated basis.
    """

    start = time.time()
    assert a.shape[0] == a.shape[1] and len(a.shape) == 2
    if b is not None:
        assert a.shape == b.shape

    if v0 is None:
        v0 = np.random.randn(a.shape[0]).astype(a.dtype)
    if b is not None:
        v0 = v0 / np.sqrt(np.vdot(v0, b@v0))
    else:
        v0 = v0 / np.linalg.norm(v0)
    
    if M is None:
        M = pyamg.aggregation.smoothed_aggregation_solver(b)
        M = M.aspreconditioner()

    #initialize states
    alpha=[]
    beta=[]

    Q = v0[:,None]

    for i in range(v):
        if i == 0:
            qi_1 = Q[:,0]
        else:
            if b is not None and not implicit:
                qi_1 = Q[:, :i+1]
            else:
                qi_1 = Q[:, i-1:i+1]
        
        qi=Q[:,i]

        z = a @ qi # a must support @ op
        ai = np.vdot(qi,z) # |z>=a_i|q_i> > a_i = <q_i|z>
        
        if b is not None and implicit: # Then all inner product need to be measured in <b> metric
            # Tz = sp.sparse.linalg.inv(b) @ z
            tmp = np.dot(qi_1.T.conj(),z)
            z, _ = sp.sparse.linalg.cg(b, z, M=M, atol=1e-8) # b^{-1} is contained implicitly in the new basis.
            # print(np.abs(z-Tz).max(), "maximum Tz")
        elif b is not None and not implicit:
            tmp = np.dot(qi_1.T.conj(),b @ z)
        else:
            tmp = np.dot(qi_1.T.conj(),z)

        tmp = np.dot(qi_1, tmp)
        z=z-tmp

        if b is not None:
            bi = np.sqrt(np.vdot(z, b @ z))
        else:
            bi = np.sqrt(np.vdot(z,z))

        alpha.append(ai)
        beta.append(bi)

        if i == v-1:
            break
        z = z*(1./bi)

        # enforce orthgonalization again
        if i > 0:
            if reorth:
                Q_i = icgs(z, Q, M=b)
            else:
                Q_i = z
        else:
            Q_i = z
            
        Q = np.append(Q,Q_i[:,None],axis=-1)

        if np.sum(abs(bi))<1e-20:
            print('Warning! bad krylov space!')
    
    return Q, alpha, beta

def blanczos(A, v0=None, B=None, v: int=30, reorth=False, return_Q=False):
    """
        Batched CG, CR/MINRES solver using lanczos iteraction with restart and selected reorthogonalization.
        The problem setting is: Ax=b, under metric B.

        The problem is solved with krylov space K(A, v0).

        Parameters:
            :a: matrix, both sparse and dense format is supported.
            :v0: the initial basis, should be 1d vector with the same length as :a:.
            :b: the positive definite matrix, defining the inner product of basis of a.
            :vr: maximum number of dimension of Krylov subspace to generate per restart.
            :mr: maximum number of restart allowed
            :reorth: orthogonalize of current vector w.r.t. all previous basis again, increase numerical 
                    stability but have additional cost.
        
        Return:
            Q: the generated basis.
            alpha: (bsz,n) elements
            beta:  (bsz,n+1) elements
    """

    # we can put all the sparse matrix multiplication as distributed MPI.
    # start = time.time()

    assert v <= A.shape[0]
    assert A.shape[0] == A.shape[1] and len(A.shape) == 2
    if B is not None:
        assert A.shape == B.shape
    
    # convert matrix type for a,b
    if isinstance(A, sp.sparse.coo_matrix):
        A = A.tocsr()
    elif isinstance(A, sparse.COO):
        A = A.to_scipy_sparse().tocsr()
    else:
        assert isinstance(A, sp.sparse.csr_matrix)
    
    if B is not None:
        if isinstance(B, sp.sparse.coo_matrix):
            B = B.tocsr()
        elif isinstance(B, sparse.COO):
            B = B.to_scipy_sparse()
            B = B.tocsr()
        else:
            assert isinstance(B, sp.sparse.csr_matrix)

    if v0 is None:
        v0 = np.random.randn(1, A.shape[0]).astype(A.dtype)
    
    if B is not None:
        bnorm = np.sqrt(np.sum(v0.conj()*dot_product_mkl(B, v0.T).T, axis=-1, keepdims=True))
    else:
        bnorm = np.linalg.norm(v0, axis=-1, keepdims=True)

    v0 = v0 / bnorm
    S = v0.shape[0]
    M = v0.shape[1]

    #initialize states
    alpha=[]
    beta=[]
    v0Q = [((bnorm * v0.conj()) * v0).sum(axis=-1)]
    bsz, matdim = v0.shape

    # adjust whether to use lu solve in scipy or use PETSc's block CG from hpddm
    if B is not None:
        Bp = csr_to_aij(B)

        if M < 1e4 or (M < 5e4 and S > 100): # The matrix is not large, it might goes up to 10^6, 10^7 would the AMG method become efficient than direct solver.
            Bc = B.tocsc()
            Bc = cholmod.cholesky(Bc)
            direct = True
        else:
            direct = False

    Q = np.asfortranarray(np.zeros((bsz, matdim, 2)).astype(v0.dtype)) # transform to fortran array so Q[:,:,x] is contiguous in Fortran

    if reorth or return_Q:
        rQ = np.asfortranarray(np.zeros((bsz, matdim, v)).astype(v0.dtype))
        rQ[:,:,0] = v0
    else:
        rQ = None
        

    Q[:,:,0] = v0
    

    # print(time.time() - start, "blanczos prepare time.")
    # start = time.time()

    for i in range(v):
        if i == 0:
            qi_1 = Q[:,:,:1]
        else:
            qi_1 = Q
        
        qi=Q[:,:,i%2] # [bsz, matdim]

        # ttime = time.time()
        z = dot_product_mkl(A, qi.T).T # a must support @ op
        # print(time.time() - ttime, "MVP time")
        # ai = (qi.conj()*z).sum(axis=-1) # |z>=a_i|q_i> > a_i = <q_i|z> # [bsz, matdim] -> [bsz]
        tmp = (qi_1.conj().transpose(0,2,1) @ z[:,:,None]).squeeze(-1) # [bsz, 1/2, matdim], [bsz, matdim,1] -> [bsz, 1/2]

        ai = tmp[:,i%2]
        
        # z_ = z.copy()
        # print(time.time() - start, " time before bsolve.")
        # start = time.time()
        if B is not None: # Then all inner product need to be measured in <b> metric
            if direct:
                z = Bc.solve_A(z.T).T.astype(B.dtype)
            else:
                z = bsolve(Bp, z.T).T.astype(B.dtype) # [bsz, matdim]
        # print(time.time() - start, " bsolve time.")
        # start = time.time()
        # print("Bsolve check: ", np.abs(B @ z.T - z_.T).max())
        tmp = (qi_1 @ tmp[:,:,None]).squeeze(-1) # [bsz,matdim]
        z=z-tmp

        # the second round of ortho
        if B is not None:
            tmp = (qi_1.conj().transpose(0,2,1) @ dot_product_mkl(B, z.T).T[:,:,None]).squeeze(-1) # [bsz, 1/2, matdim], [bsz, matdim,1] -> [bsz, 1/2]
        else:
            tmp = (qi_1.conj().transpose(0,2,1) @ z[:,:,None]).squeeze(-1)

        ai += tmp[:, i%2]

        tmp = (qi_1 @ tmp[:,:,None]).squeeze(-1) # [bsz,matdim]
        z=z-tmp

        if B is not None:
            bi = np.sqrt(np.sum(z.conj() * dot_product_mkl(B, z.T).T, axis=-1))
        else:
            bi = np.linalg.norm(z, axis=-1)

        alpha.append(ai)
        beta.append(bi)

        if i == v-1:
            break
        z = z*(1./bi[:,None])

        # enforce orthgonalization again
        if i > 0:
            if reorth:
                Q_i = bicgs(z, rQ[:,:,:i+1], M=B)
            else:
                Q_i = z
        else:
            Q_i = z
        
        Q[:,:,(i+1)%2] = Q_i
        
        if return_Q or reorth:
            rQ[:,:,i+1] = Q_i

        v0Q.append(((bnorm * v0.conj()) * Q_i).sum(axis=-1))

        # print(time.time() - start, "time after bsolve.")
        # start = time.time()

        if np.max(abs(bi))<1e-20:
            print('Warning! bad krylov space!')
    
    return np.stack(v0Q, axis=-1), np.stack(alpha, axis=-1), np.stack(beta, axis=-1), rQ # [v, S]

def bCGCR(A, v0=None, B=None, vr: int=30, mr: int=10, reorth=True):
    """
        Batched CG, CR/MINRES solver using lanczos iteraction with restart and selected reorthogonalization.
        The problem setting is: Ax=b, under metric B.

        The problem is solved with krylov space K(A, v0).

        Parameters:
            :a: matrix, both sparse and dense format is supported.
            :v0: the initial basis, should be 1d vector with the same length as :a:.
            :b: the positive definite matrix, defining the inner product of basis of a.
            :vr: maximum number of dimension of Krylov subspace to generate per restart.
            :mr: maximum number of restart allowed
            :reorth: orthogonalize of current vector w.r.t. all previous basis again, increase numerical 
                    stability but have additional cost.
        
        Return:
            Q: the generated basis.
            alpha: (bsz,n,n) elements
            beta:  (bsz,n+1,n) elements
    """

    assert A.shape[0] == A.shape[1] and len(A.shape) == 2
    if B is not None:
        assert A.shape == B.shape
    
    # convert matrix type for a,b
    if isinstance(A, sp.sparse.coo_matrix):
        A = A.tocsr()
    elif isinstance(A, sparse.COO):
        A = A.to_scipy_sparse().tocsr()
    else:
        assert isinstance(A, sp.sparse.csr_matrix)
    
    if B is not None:
        if isinstance(B, sp.sparse.coo_matrix):
            B = B.tocsr()
        elif isinstance(B, sparse.COO):
            B = B.to_scipy_sparse()
            B = B.tocsr()
        else:
            assert isinstance(B, sp.sparse.csr_matrix)

    if v0 is None:
        v0 = np.random.randn(1, A.shape[0]).astype(A.dtype)
    
    if B is not None:
        bnorm = np.sqrt(np.sum(v0.conj()*(B@v0.T).T, axis=-1, keepdims=True))
    else:
        bnorm = np.linalg.norm(v0, axis=-1, keepdims=True)

    v0 = v0 / bnorm

    #initialize states
    alpha=[]
    beta=[]
    v0Q = []
    bsz, matdim = v0.shape

    for ri in range(mr):
        alpha.append([]); beta.append([]); v0Q.append([])
        Q = np.zeros((bsz, matdim, 2)).astype(v0.dtype)
        Q[:,:,0] = v0

        # res: initialize the residual for convergence check
        res = 0
        for i in range(vr):
            if i == 0:
                qi_1 = Q[:,:,:1]
            else:
                qi_1 = Q
            
            qi=Q[:,:,i] # [bsz, matdim]

            z = (A @ qi.T).T # a must support @ op
            ai = (qi.conj()*z).sum(axis=-1) # |z>=a_i|q_i> > a_i = <q_i|z> # [bsz, matdim] -> [bsz]
            tmp = (qi_1.conj().transpose(0,2,1) @ z[:,:,None]).squeeze(-1) # [bsz, 1/2, matdim], [bsz, matdim,1] -> [bsz, 1/2]
            
            if B is not None: # Then all inner product need to be measured in <b> metric
                z = bsolve(B, z.T).T # [bsz, matdim]

            tmp = (qi_1 @ tmp[:,:,None]).squeeze(-1) # [bsz,matdim]
            z=z-tmp

            if B is not None:
                bi = np.sqrt(np.sum(z.conj() * (B @ z.T).T, axis=-1))
            else:
                bi = np.linalg.norm(z.conj() * z, axis=-1)

            alpha[ri].append(ai)
            beta[ri].append(bi)

            if i == vr-1:
                break
            z = z*(1./bi[:,None])

            # enforce orthgonalization again
            if i > 0:
                if reorth:
                    Q_i = bicgs(z, Q, M=B)
                else:
                    Q_i = z
            else:
                Q_i = z
            
            Q[:,:,(i+1)%2] = Q_i
            v0Q[ri].append(bnorm * (v0 * Q_i).sum(axis=-1, keepdim=True))

            if np.max(abs(bi))<1e-20:
                print('Warning! bad krylov space!')
        
        v0 = 0 # update the new approximate
        alpha[ri] = np.stack(alpha[ri], axis=1); beta[ri] = np.stack(beta[ri], axis=1); v0Q[ri] = np.stack(v0Q[ri], axis=-1)
    
    return v0Q, alpha, beta



# def step_Krylov_Arnoldi(a, Q=None, b=None, M=None):
#     assert a.shape[0] == a.shape[1] and len(a.shape) == 2
#     if b is not None:
#         assert a.shape == b.shape

#     if Q is None:
#         Q = np.random.randn(a.shape[0]).astype(a.dtype)
#         Q = Q / np.linalg.norm(Q)
#         Q = Q[:, None]
#     elif Q.ndim == 1:
#         Q = Q / np.linalg.norm(Q)
#         Q = Q[:, None]
#     elif Q.ndim == 2 and Q.shape[1] == 1:
#         Q = Q / np.linalg.norm(Q)
#     else:
#         assert Q.ndim == 2
    
#     if M is None:
#         M = pyamg.aggregation.smoothed_aggregation_solver(b)
#         M = M.aspreconditioner()

#     #initialize states

#     i = Q.shape[1] - 1

#     if i == 0:
#         qi_1 = Q[:,0]
#     else:
#         qi_1 = Q[:, i-1:i+1]
    
#     qi=Q[:,i]

#     z = a.dot(qi)
#     if b is not None:
#         z, _ = sp.sparse.linalg.cg(b, z, M=M)
#     tmp = np.dot(qi_1.T.conj(),z)
#     tmp = np.dot(qi_1, tmp)
#     z=z-tmp

#     bi = np.sqrt(np.dot(z.conj(),z))

#     z = z*(1./bi)
#     Q_i = icgs(z[:,None],Q)
#     Q = np.append(Q,Q_i,axis=-1)

#     if np.sum(abs(bi))<1e-20:
#         print('Warning! bad krylov space!')
    
#     return Q


# def batched_step_Krylov_Arnoldi(a, Q=None, b=None):
#     assert a.shape[0] == a.shape[1] and len(a.shape) == 2
#     if b is not None:
#         assert a.shape == b.shape

#     if Q is None:
#         Q = np.random.randn(1, a.shape[0],1).astype(a.dtype)
#         Q = Q / np.linalg.norm(Q)
#     elif Q.ndim == 1:
#         Q = Q / np.linalg.norm(Q)
#         Q = Q[None,:,None]
#     elif Q.ndim == 2 and Q.shape[1] == 1:
#         Q = Q / np.linalg.norm(Q)
#         Q = Q[None,:,:]
#     elif Q.ndim == 2 and Q.shape[0] == 1:
#         Q = Q / np.linalg.norm(Q)
#         Q = Q[:,:,None]
#     else:
#         assert Q.ndim == 3 # [bsz, matdim, Krdim]

#     #initialize states

#     i = Q.shape[2] - 1

#     if i == 0:
#         qi_1 = Q[:,:,0]
#     else:
#         qi_1 = Q[:,:,i-1:i+1]
    
#     qi=Q[:,:,i] # [bsz, matdim]

#     z = (a @ qi.T).T # [bsz, matdim]
#     if b is not None:
#         z = bsolve(b, z.T).T # [bsz, matdim]
#     tmp = qi_1.conj().transpose(0,2,1) @ z[:,:,None] # [bsz, 2, matdim] x [bsz, matdim, 1] -> [bsz, 2, 1]
#     tmp = qi_1 @ tmp # [bsz, matdim, 2] x [bsz, 2, 1] -> [bsz, matdim, 1]
#     z=z-tmp[:,:,0] # [bsz, matdim]

#     bi = np.sqrt((z.conj()*z).sum(axis=-1)) # [bsz]

#     z = z*(1./bi[:,None]) # [bsz, matdim]
#     Q_i = icgs(z[:,:,None],Q)
#     Q = np.append(Q,Q_i,axis=-1)

#     if np.sum(abs(bi))<1e-20:
#         print('Warning! bad krylov space!')
    
#     return Q
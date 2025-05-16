from src.linalg.dcgmrhs import cgmrhs
import scipy.sparse as sp
import numpy as np

A = np.random.randn(10,10)
A = A + A.T
eigs = np.linalg.eigvalsh(A)
print(eigs)
A = A - (eigs[0] - 1) * np.eye(A.shape[0])
A = sp.csr_array(A)
b = np.random.randn(10,10)
    
X = cgmrhs(
    A=A,
    b=b,
    Atype="symmetric"
)

print(X[0])


print(np.linalg.inv(A.todense()) @ b-X[0].reshape(b.shape))
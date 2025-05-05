import numpy as np

def float2comlex(dtype):
    if isinstance(dtype, str):
        dtype =  getattr(np, dtype)
    
    if dtype is np.float32:
        cdtype = np.complex64
    elif dtype is np.float64:
        cdtype = np.complex128
    else:
        raise ValueError("the dtype is not supported! now only float64, float32 is supported!")
    return cdtype

def float2complex(A):
    if A.dtype == np.float32:
        A = A.astype(np.complex64)
    elif A.dtype == np.float64:
        A = A.astype(np.complex128)
    elif A.dtype == np.float128:
        A = A.astype(np.complex256)
    else:
        raise TypeError("The A should have numpy float dtype")
    
    return A

def real_csr_to_complex(av_real):
    """
    ai, aj  : int32 CSR index arrays  (already valid)
    av_real : float64 data            (imag part == 0)
    returns : complex128 data         (shared index arrays)
    """
    av_cplx = np.empty_like(av_real, dtype=np.complex128)   # allocate once
    av_cplx.real = av_real                                  # vectorised copy
    av_cplx.imag = 0.0                                      # fill in-place
    return av_cplx

def real_dense_to_complex(b):
    if np.iscomplexobj(b):
        return np.asfortranarray(b)        # ensure F-contiguous
    z = np.empty(b.shape, np.complex128, order='F')
    z.real = b
    z.imag = 0.0
    return z

def complex2float(A):
    if A.dtype == np.complex64:
        A = A.astype(np.float32)
    elif A.dtype == np.complex128:
        A = A.astype(np.float64)
    elif A.dtype == np.complex256:
        A = A.astype(np.float128)
    else:
        raise TypeError("The A should have numpy complex dtype")
    
    return A
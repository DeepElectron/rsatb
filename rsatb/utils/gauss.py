import numpy as np
from typing import Union

def gauss_xw(xl: Union[np.ndarray, float, int], xu: Union[np.ndarray, float, int], n: int=100):
    '''calculates the Gauss-Legendre quadrature points and weights for numerical  integration.
    
    Parameters
    ----------
    xl
        The lower limit of integration.
    xu
        The upper limit of the integration range.
    n, optional
        the number of points to be used in the Gauss-Legendre quadrature.    
    Returns
    -------
        point xs and their weights wlg.
    
    '''
    if isinstance(xu, (int, float)):
        ndim = 0
        assert isinstance(xl, (int, float))
    else:
        ndim = len(xu.shape)

    xlg, wlg = np.polynomial.legendre.leggauss(n)
    shape = xlg.shape
    xlg = xlg.reshape(shape + (None,) * ndim)  # (n, *nx)
    wlg = wlg.reshape(shape + (None,) * ndim)  # (n, *nx)
    wlg *= 0.5 * (xu - xl)
    xs = xlg * (0.5 * (xu - xl)) + (0.5 * (xu + xl))  # (n, *nx)

    return xs, wlg

def tanh_x(Emin: Union[np.ndarray, float, int], Emax: Union[np.ndarray, float, int], M: int=100, h: float=0.1):
    Emid = (Emin + Emax) / 2
    xl, xr = - M * h, M * h
    xs = np.arange(xl, xr, h)
    xs = np.append(xs, xr)
    xs = ((Emax-Emin) / 2) * np.sinh((np.pi / 2) * np.sinh(xs)) + Emid

    return xs
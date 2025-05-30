from sparse_dot_mkl._mkl_interface import (
    MKL,
    sparse_matrix_t,
    SPARSE_DIAG_NON_UNIT,
    SPARSE_MATRIX_TYPE_SYMMETRIC,
    SPARSE_MATRIX_TYPE_HERMITIAN,
    SPARSE_MATRIX_TYPE_GENERAL,
    SPARSE_FILL_MODE_FULL,
    SPARSE_FILL_MODE_LOWER,
    SPARSE_FILL_MODE_UPPER,
    SPARSE_DIAG_UNIT,
    _mkl_scalar,
    SPARSE_OPERATION_NON_TRANSPOSE
)

from numpy.ctypeslib import ndpointer

from ._iss import (
    MKLIterativeSparseSolver,
    _create_empty_temp_arrays,
    ConvergenceWarning,
    DEFAULT_MAX_ITER
)
import numpy as _np
import ctypes as _ctypes
import warnings

_mkl_sp_mv_funcs = {
    (False, False): MKL._mkl_sparse_s_mv,
    (True, False): MKL._mkl_sparse_d_mv,
    (False, True): MKL._mkl_sparse_c_mv,
    (True, True): MKL._mkl_sparse_z_mv,
}

_args_ft = [
            _ctypes.POINTER(MKL.MKL_INT),
            ndpointer(dtype=_ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'),
            _ctypes.POINTER(MKL.MKL_INT),
            ndpointer(dtype=_ctypes.c_double, ndim=1, flags='C_CONTIGUOUS'),
            _ctypes.POINTER(MKL.MKL_INT),
            ndpointer(dtype=MKL.MKL_INT, flags='C_CONTIGUOUS'),

            ndpointer(dtype=_ctypes.c_double, flags='C_CONTIGUOUS'),
            ndpointer(dtype=_ctypes.c_double, flags='C_CONTIGUOUS')
        ]

_args_tt = [
            _ctypes.POINTER(MKL.MKL_INT),
            ndpointer(dtype=_np.cdouble, ndim=1, flags='C_CONTIGUOUS'),
            _ctypes.POINTER(MKL.MKL_INT),
            ndpointer(dtype=_np.cdouble, ndim=1, flags='C_CONTIGUOUS'),
            _ctypes.POINTER(MKL.MKL_INT),
            ndpointer(dtype=MKL.MKL_INT, flags='C_CONTIGUOUS'),

            ndpointer(dtype=_ctypes.c_double, flags='C_CONTIGUOUS'),
            ndpointer(dtype=_np.cdouble, flags='C_CONTIGUOUS')
        ]

MKL._dcgmrhs_init.argtypes = _args_ft[:4] + [_ctypes.POINTER(MKL.MKL_INT)] + _args_ft[4:]
MKL._dcgmrhs_init.restype = None

MKL._dcgmrhs_check.argtypes = _args_ft
MKL._dcgmrhs_check.restype = None

# MKL._dcgmrhs.argtypes = _args[:4] + [ndpointer(dtype=_ctypes.c_double, flags='C_CONTIGUOUS')] + \
#     [_args[4]] + [ndpointer(dtype=_ctypes.c_double, flags='C_CONTIGUOUS')] + _args[5:]
# MKL._dcgmrhs_check.restype = None
MKL._dcgmrhs.argtypes = _args_ft
MKL._dcgmrhs.restype = None

MKL._dcgmrhs_get.argtypes = _args_ft + [_ctypes.POINTER(MKL.MKL_INT)]

MKL._dcgmrhs_get.restype = None


def dcgmrhs_init(
    n,
    b,
    nrhs=None,
    x=None,
    ipar=None,
    dpar=None,
    tmp=None
):
    """
    CG solver initialization.
    If x, ipar, dpar, and tmp are not passed, will be initalized
    with standard values.

    :param n: Number of predictor variables to model
    :type n: int
    :param b: Right-hand vector b
    :type b: np.ndarray

    :param x: Solution vector x, defaults to None
    :type x: np.ndarray, optional
    :param ipar: Integer parameter vector with settings for solver,
        defaults to None
    :type ipar: np.ndarray, optional
    :param dpar: Double parameter vector with settings for solver,
        defaults to None
    :type dpar: np.ndarray, optional
    :param tmp: Temp vector for solver computations, defaults to None
    :type tmp: np.ndarray, optional

    :return x: Solution vector x
    :rtype x: np.ndarray
    :return ipar: Integer parameter vector with settings for solver
    :type ipar: np.ndarray
    :return dpar: Double parameter vector with settings for solver
    :type dpar: np.ndarray
    :return tmp: Temp vector for solver computations
    :type tmp: np.ndarray
    """

    if x is None:
        x = _np.zeros((n*nrhs), dtype=_ctypes.c_double)
        x = x.ravel("F")
    else:
        x = x.ravel("F")

    
    ipar, dpar, tmp = _create_empty_temp_arrays(ipar, dpar, tmp, nrhs, (4+nrhs) * n)
    ipar = ipar.ravel("F")
    dpar = dpar.ravel("F")
    tmp = tmp.ravel("F")
    # tmp = tmp.reshape((n,4+nrhs), order="F")
    # tmp = _np.ascontiguousarray(tmp)
    # print(tmp.flags)

    ret_val = MKL.MKL_INT()

    MKL._dcgmrhs_init(
        _ctypes.byref(MKL.MKL_INT(n)),
        x,
        _ctypes.byref(MKL.MKL_INT(nrhs)),
        b,
        _ctypes.byref(MKL.MKL_INT(1)),
        _ctypes.byref(ret_val),
        ipar,
        dpar,
        tmp
    )

    status = ret_val.value

    if status == -10000:
        raise RuntimeError(
            "dcgmrhs_init returned -10000 (Failed to complete task))"
        )
    elif status != 0 :
        raise RuntimeError(
            f"dcgmrhs_init returned {status} (Unknown error))"
        )
    
    # ipar[2] = nrhs

    # print(ipar)

    return x, ipar, dpar, tmp


def dcgmrhs_check(
    n,
    x,
    nrhs,
    b,
    ipar,
    dpar,
    tmp
):
    """
    CG solver checker. 
    Verifies that the initialized parameters are valid.

    :param n: Number of predictor variables to model
    :type n: int
    :param b: Right-hand vector b
    :type b: np.ndarray
    :param x: Solution vector x
    :type x: np.ndarray
    :param ipar: Integer parameter vector with settings for solver
    :type ipar: np.ndarray
    :param dpar: Double parameter vector with settings for solver
    :type dpar: np.ndarray
    :param tmp: Temp vector for solver computations
    :type tmp: np.ndarray

    :return ret_val: Returned status of the check
    :rtype ret_val: int
    """

    ret_val = MKL.MKL_INT()

    MKL._dcgmrhs_check(
        _ctypes.byref(MKL.MKL_INT(n)),
        x,
        _ctypes.byref(MKL.MKL_INT(nrhs)),
        b,
        _ctypes.byref(ret_val),
        ipar,
        dpar,
        tmp
    )

    status = ret_val.value

    if status == -1100:
        raise RuntimeError("dcgmrhs_check returned -1100 (operation failed)")
    elif status == -1001 or status == -1011:
        warnings.warn(
            f"dcgmrhs_check raised warnings: {status}",
            RuntimeWarning
        )

    return status

def dcgmrhs(n, x, nrhs, b, ipar, dpar, tmp):
    """
    Execute CG solver iteration

    :param n: Number of predictor variables to model
    :type n: int
    :param b: Right-hand vector b
    :type b: np.ndarray
    :param x: Solution vector x
    :type x: np.ndarray
    :param ipar: Integer parameter vector with settings for solver
    :type ipar: np.ndarray
    :param dpar: Double parameter vector with settings for solver
    :type dpar: np.ndarray
    :param tmp: Temp vector for solver computations
    :type tmp: np.ndarray

    :return ret_val: Returned status of the check
    :rtype ret_val: int
    """
    
    ret_val = MKL.MKL_INT()

    MKL._dcgmrhs(
        _ctypes.byref(MKL.MKL_INT(n)),
        x,
        _ctypes.byref(MKL.MKL_INT(nrhs)),
        b,
        _ctypes.byref(ret_val),
        ipar,
        dpar,
        tmp
    )

    status = ret_val.value

    if status == -2:
        raise RuntimeError(
            "dcgmrhs returned -2 (Divide by zero. This situation happens if "
            "the matrix is non-positive definite or almost non-positive "
            "definite)"
        )
    elif status == -10:
        raise RuntimeError(
            "dcgmrhs returned -10 (The residual norm is invalid. This usually "
            "happens because the value dpar(6) was altered outside of the "
            "routine, or the dcg_check routine was not called.)"
        )
    elif status == -11:
        raise RuntimeError(
            "dcgmrhs returned -11 (Infinite cycle. This usually happens because "
            "the values ipar(8), ipar(9), ipar(10) were altered outside of "
            "the routine, or the dcg_check routine was not called.)"
        )
    elif status < 0:
        raise RuntimeError(
            f"dcgmrhs returned {status} (Unknown error)"
        )

    return status

def dcgmrhs_get(
    n,
    x,
    nrhs,
    b,
    ipar,
    dpar,
    tmp
):
    """
    CG solver checker. 
    Verifies that the initialized parameters are valid.

    :param n: Number of predictor variables to model
    :type n: int
    :param b: Right-hand vector b
    :type b: np.ndarray
    :param x: Solution vector x
    :type x: np.ndarray
    :param ipar: Integer parameter vector with settings for solver
    :type ipar: np.ndarray
    :param dpar: Double parameter vector with settings for solver
    :type dpar: np.ndarray
    :param tmp: Temp vector for solver computations
    :type tmp: np.ndarray

    :return ret_val: Returned status of the check
    :rtype ret_val: int
    """

    ret_val, iter_count = MKL.MKL_INT(), MKL.MKL_INT()

    MKL._dcgmrhs_get(
        _ctypes.byref(MKL.MKL_INT(n)),
        x,
        _ctypes.byref(MKL.MKL_INT(nrhs)),
        b,
        _ctypes.byref(ret_val),
        ipar,
        dpar,
        tmp,
        _ctypes.byref(iter_count)
    )

    status = ret_val.value
    num_iteration = iter_count.value

    if status == -10000:
        raise RuntimeError(
            "dfgmres_get returned -10000 (Failed to complete task))"
        )
    elif status == -12:
        raise RuntimeError(
            "dfgmres_get returned -12 (The routine was interrupted because "
            "errors were found in the method parameters. Usually this "
            "happens if the parameters ipar and dpar were altered by mistake "
            "outside the routine.)"
        )
    elif status != 0 :
        raise RuntimeError(
            f"dfgmres_get returned {status} (Unknown error))"
        )

    return num_iteration


class CGMRHSIterativeSparseSolver(MKLIterativeSparseSolver):

    def initialize_solver(self):
        if self.cplx and self.dbl:
            MKL._dcgmrhs_init.argtypes = _args_tt[:4] + [_ctypes.POINTER(MKL.MKL_INT)] + _args_tt[4:]
            MKL._dcgmrhs_check.argtypes = _args_tt
            MKL._dcgmrhs.argtypes = _args_tt
            MKL._dcgmrhs_get.argtypes = _args_tt + [_ctypes.POINTER(MKL.MKL_INT)]

        self.x, self.ipar, self.dpar, self.tmp = dcgmrhs_init(
            self.n,
            self.b,
            self.nrhs,
            self.x,
            self.ipar,
            self.dpar,
            self.tmp
        )

    def check_solver(self):
        return dcgmrhs_check(
            self.n,
            self.x,
            self.nrhs,
            self.b,
            self.ipar,
            self.dpar,
            self.tmp
        )

    def solve_iteration(self):
        return dcgmrhs(
            self.n,
            self.x,
            self.nrhs,
            self.b,
            self.ipar,
            self.dpar,
            self.tmp
        )

    def get_solution(self):
        dcgmrhs_get(
            self.n,
            self.x,
            self.nrhs,
            self.b,
            self.ipar,
            self.dpar,
            self.tmp
        )
        return self.x

    def solve(self):
        """
        Run solver until complete (tolerances are reached)
        or until max_iter is reached.

        Raises a warning if the solution did not converge

        :return: Solution vector x
        :rtype: np.ndarray
        """

        for return_value in self:

            self.final_code = return_value

            if return_value == 1:
                """multiply the matrix by tmp [0:n - 1), put the result in tmp[n:2*n - 1), 
                and return the control to the dcg/dcgmrhs routine;"""
                # print("Step1")
                pass # this should not been handelled since ISS class inplement next as update_tmp
            elif return_value == 2:
                """
                to perform the stopping tests. If they fail, return the control to the dcg/dcgmrhs routine. 
                If the stopping tests succeed, it indicates that the solution is found and stored in the x array;
                """
                print("Step2")
                pass
            elif return_value == 3:
                """
                for SRHS: apply the preconditioner to tmp[2*n:3*n - 1], put the result in tmp[3*n:4*n - 1], 
                and return the control to the dcg routine;

                for MRHS: apply the preconditioner to tmp[(2+ipar[2])*n:(3 + ipar[2])*n - 1], put the result in tmp[(3 + ipar[2])*n:(4 + ipar[2])*n - 1], 
                and return the control to the dcgmrhs routine.
                """

                # j = int(self.ipar[2])
                # self.tmp[3 + j, :] = self.tmp[2 + j, :]
                # self.tmp[(2+j)*self.n:(3+j)*self.n] = self.tmp[(3+j)*self.n:(4+j)*self.n]
                print("Step3")
                
            elif return_value == 0:
                # self.get_solution()
                break

            if self.current_iter > self.max_iter:
                warnings.warn(
                    f"Solution did not converge in {self.current_iter} iterations",
                    ConvergenceWarning
                )
                break

        return self.x

    def set_initial_parameters(self):

        # Set max iterations in ipar[5]
        self.ipar[4] = self.max_iter

        # Suppress messages unless MKL debug is on or verbose is True
        self.ipar[5] = 1 if MKL.MKL_DEBUG or self.verbose else 0
        self.ipar[6] = 1 if MKL.MKL_DEBUG or self.verbose else 0

        # Set automatic stopping tests
        self.ipar[7] = 1
        self.ipar[8] = 1
        self.ipar[9] = 0

        if self.M is None:
            self.ipar[10] = 0

        # Set tolerances
        self.dpar[0] = self.r_tol
        self.dpar[1] = self.a_tol

    def update_tmp(self):
        alpha = _mkl_scalar(1., self.cplx, self.dbl)
        beta = _mkl_scalar(0., self.cplx, self.dbl)

        _mkl_sp_mv_funcs[(self.dbl, self.cplx)](
            SPARSE_OPERATION_NON_TRANSPOSE,
            alpha,
            self.A,
            self.matrix_A_descr,
            self.tmp[:self.n],
            beta,
            self.tmp[self.n:2*self.n],
        )

            # self.solve_iteration()

        # print(self.a_tol, self.r_tol)
        # print(self.tmp[:].reshape(-1,self.n))
        print(_np.linalg.norm(self.tmp[:].reshape(-1,self.n), axis=-1))

        # _mkl_sp_mv_funcs[(self.dbl, self.cplx)](
        #     SPARSE_OPERATION_NON_TRANSPOSE,
        #     alpha,
        #     self.A,
        #     self.matrix_A_descr,
        #     self.tmp[4*self.n:5*self.n],
        #     beta,
        #     self.tmp[5*self.n:6*self.n],
        # )

        # print(self.tmp[self.n*5:self.n*6])


def cgmrhs(
    A,
    b,
    x0=None,
    tol=1e-09,
    maxiter=DEFAULT_MAX_ITER,
    M=None,
    Atype="general",
    unit=True,
    callback=None,
    atol=1e-9
):
    """
    Call Conjugate Gradient iterative solver and iterate without user input

    :param A: Scipy sparse matrix
    :type A: sp.sparse.csr_matrix
    :param b: Dense RHS matrix (N, ) or (N, 1)
    :type b: np.ndarray
    :param x0: Starting guess for the solution,
        defaults to None
    :type x0: np.ndarray, optional
    :param tol: Tolerances for convergence, defaults to 1e-05
    :type tol: float, optional
    :param maxiter: Maximum number of iterations, defaults to None
    :type maxiter: int, optional
    :param M: Preconditioner for A, defaults to None
    :type M: np.ndarray, optional
    :param callback: User-supplied function to call after each iteration,
        defaults to None
    :type callback: function, optional
    :param atol: Tolerances for convergence, defaults to None
    :type atol: float, optional
    """

    if M is not None:
        raise NotImplementedError("Preconditioner M not supported")
    if callback is not None:
        raise NotImplementedError("callback is not supported")
    
    if isinstance(A, sparse_matrix_t):
        cplx = bool(_np.iscomplex(b[0,0]))
        if b.dtype in [_np.cdouble, _np.double]:
            dbl = True
        else:
            dbl=False
    else:
        cplx = None,
        dbl = None
    
    if isinstance(Atype, _ctypes.Structure):
        direct = True
    else:
        direct = False
        if Atype == "hermitian":
            _Atype = SPARSE_MATRIX_TYPE_HERMITIAN
            Ftype = SPARSE_FILL_MODE_UPPER
            if unit:
                Dtype = SPARSE_DIAG_UNIT
            else:
                Dtype = SPARSE_DIAG_NON_UNIT
        elif Atype == "symmetric":
            _Atype = SPARSE_MATRIX_TYPE_SYMMETRIC
            Ftype = SPARSE_FILL_MODE_UPPER
            if unit:
                Dtype = SPARSE_DIAG_UNIT
            else:
                Dtype = SPARSE_DIAG_NON_UNIT
        elif Atype == "general":
            _Atype = SPARSE_MATRIX_TYPE_GENERAL
            Ftype = SPARSE_FILL_MODE_FULL
            Dtype = SPARSE_DIAG_NON_UNIT
        else:
            raise ValueError("Atype should be among ['hermitian', 'symmetric', 'general'].")


    with CGMRHSIterativeSparseSolver(
        A,
        b,
        n=b.shape[0],
        x=x0,
        verbose=False,
        max_iter=maxiter,
        a_tol=atol,
        r_tol=tol,
        cplx=cplx,
        dbl=dbl,
    ) as cg_solver:
        
        if direct:
            cg_solver.matrix_A_descr = Atype
        else:
            cg_solver.set_sparse_matrix_descr(
                sparse_matrix_type_t=_Atype,
                sparse_fill_mode_t=Ftype,
                sparse_diag_type_t=Dtype
                )
        try:
            x = cg_solver.solve()
        except RuntimeError:
            RuntimeWarning("There is error during the run!")
            return cg_solver.x, cg_solver.final_code
        
        return x, cg_solver.final_code
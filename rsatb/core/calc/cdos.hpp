#include <petsc.h>
#include "cheby.hpp"

namespace rsatb::calc{

template<typename T>
void corr_dos(Mat H, KSP ksp, T Psi_in, pet double Escale, double dE, int nrhs, double **cdos, int *nt){
    /*
        1. Create Bessel grid and time grid;
        2. Do orbital selection according if computing pdos
        3. perform propergation, record the cdos
    */
    *nt = ceil(Escale / dE);
    double dt = pi;
    PetscComplex *coeff;
    PetscComplex _cdos_[2*(*nt)+1]={0};
    _cdos_[*nt] = 1;

    timeprop_coeff(dt, 1e-14, true, &coeff, int *n_bes);

    /* Skip orbital selection currently */

    if(nrhs==1){
        Vec  buff[3], Psi_out, Psi_curr;
        VecDuplicate(Psi_in, buff);
        VecDuplicate(Psi_in, buff+1);
        VecDuplicate(Psi_in, buff+2);
        VecDuplicate(Psi_in, &Psi_out);
        VecDuplicate(Psi_in, &Psi_curr);
        VecCopy(Psi_in, Psi_curr);


        for(int it=0; it<nt; ++it){
            cheby_recur(
                H, ksp, Psi_curr,
                buff, // length 3 vec
                coeff, n_bes, 
                Psi_out, true
                );
            VecDot(Psi_in, Psi_out, _cdos_ + (*nt+it+1));
            VecCopy(Psi_out, Psi_curr);
        }

        VecDestroy(buff); VecDestroy(buff+1); VecDestroy(buff+2);
        VecDestroy(&Psi_out); VecDestroy(&Psi_curr);

    }else if(nrhs>1)
    {
        Mat buff[4], Psi_out, Psi_curr;
        Vec v_psi_in, v_psi_out;
        PetScalar temp=0;
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff+1);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff+2);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff+3);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &Psi_out);
        MatDuplicate(Psi_in, MAT_COPY_VALUES, &Psi_curr);

        for(int it=0; it<nt; ++it){
            cheby_recur(
                H, ksp, Psi_curr,
                buff, // length 3 vec
                coeff, n_bes,
                Psi_out, true
                );
            for(int in=0; in<nrhs; ++in){
                MatDenseGetColumnVecRead(Psi_in, j, &v_psi_in);
                MatDenseGetColumnVecRead(Psi_out, j, &v_psi_out);
                VecDot(v_psi_in, v_psi_out, &temp);
                _cdos_[*nt+it+1] += temp;
                MatDenseRestoreColumnVecRead(A, j, &aj);
                MatDenseRestoreColumnVecRead(B, j, &bj);
            }
            _cdos_[*nt+it+1] /= nrhs;
            MatCopy(Psi_out, Psi_curr, SAME_NONZERO_PATTERN);
        }

        MatDestroy(buff); MatDestroy(buff+1); MatDestroy(buff+2); MatDestroy(buff+3);
        MatDestroy(&Psi_out); MatDestroy(&Psi_curr);
        VecDestroy(&v_psi_in); VecDestroy(&v_psi_out);

    }else{
        throw std::invalid_argument("The number of RHS should be always larger than 0!");
    }
}
}
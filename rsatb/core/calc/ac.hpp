#ifndef AC_CALC
#define AC_CALC

#include <petsc.h>
#include "cheby.hpp"
#include "fermi_dirac.hpp"
#include <math.h>

namespace rsatb::calc{

void current(Mat H, KSP ksp, Mat Rx, Mat Ry, Mat Rz, Vec Psi_in, Vec Psix, Vec Psiy, Vec Psiz, Vec buff1, Vec buff2){ // apply iS^{-1}[H, R] on Psi_in
    // -i(e/\hbar)[\hat{H},\hat{R}] = iSinv(-HSinvR + RSinvH) -> -iSinv[R(-SinvHx) + HSinvRx] use atomic unit
    MatMult(H, Psi_in, buff2);
    KSPSolve(ksp, buff2, buff2);
    VecScale(buff2, -1.0);

    // x
    MatMult(Rx, Psi_in, buff1);
    KSPSolve(ksp, buff1, buff1);
    MatMult(H, buff1, Psix);
    MatMultAdd(Rx, buff2, Psix, Psix);

    // y
    MatMult(Ry, Psi_in, buff1);
    KSPSolve(ksp, buff1, buff1);
    MatMult(H, buff1, Psiy);
    MatMultAdd(Ry, buff2, Psiy, Psiy);

    // z
    MatMult(Rz, Psi_in, buff1);
    KSPSolve(ksp, buff1, buff1);
    MatMult(H, buff1, Psiz);
    MatMultAdd(Rz, buff2, Psiz, Psiz);

    // S^{-1} solve
    KSPSolve(ksp, Psix, Psix);
    KSPSolve(ksp, Psiy, Psiy);
    KSPSolve(ksp, Psiz, Psiz);

    const PetscComplex  nI(0.0, -1.0);

    VecScale(Psix, nI);
    VecScale(Psiy, nI);
    VecScale(Psiz, nI);
}

void current_adjoint(Mat H, KSP ksp, Mat Rx, Mat Ry, Mat Rz, Vec Psi_in, Vec Psix, Vec Psiy, Vec Psiz, Vec buff1, Vec buff2, Vec buff3){ // apply [R, H]S^{-1} on Psi_in
    // (-i[\hat{H},\hat{R}])* = i(RSinvH - HSinvR)Sinv -> -i[R(-SinvHSinvx) + HSinvRSinvx]
    KSPSolve(ksp, Psi_in, buff3);
    MatMult(H, buff3, buff1);
    KSPSolve(ksp, buff1, buff2); // buff2 = SinvHSinvx
    VecScale(buff2, -1.0);
    

    // x
    MatMult(Rx, buff3, Psix);
    KSPSolve(ksp, Psix, buff1);
    MatMult(H, buff1, Psix);
    MatMultAdd(Rx, buff2, Psix, Psix);

    // y
    MatMult(Ry, buff3, Psiy);
    KSPSolve(ksp, Psiy, buff1);
    MatMult(H, buff1, Psiy);
    MatMultAdd(Ry, buff2, Psiy, Psiy);

    // z
    MatMult(Rz, buff3, Psiz);
    KSPSolve(ksp, Psiz, buff1);
    MatMult(H, buff1, Psiz);
    MatMultAdd(Rz, buff2, Psiz, Psiz);

    const PetscComplex  nI(0.0, -1.0);

    VecScale(Psix, nI);
    VecScale(Psiy, nI);
    VecScale(Psiz, nI);

}

void current_mult(Mat H, KSP ksp, Mat Rx, Mat Ry, Mat Rz, Mat *Psi_in, Mat *Psix, Mat *Psiy, Mat *Psiz, Mat *buff1, Mat *buff2){ // apply iS^{-1}[H, R] on Psi_in
    // -i[\hat{H},\hat{R}] = -iSinv(HSinvR - RSinvH) -> -iSinv[R(-SinvHx) + HSinvRx]
    MatMatMult(H, *Psi_in, MAT_REUSE_MATRIX, PETSC_CURRENT, buff1);
    KSPMatSolve(ksp, *buff1, *buff2); // buff2 = SinvH|in>

    // x
    MatMatMult(Rx, *Psi_in, MAT_REUSE_MATRIX, PETSC_CURRENT, buff1); // buff1 = Rx|in>
    KSPMatSolve(ksp, *buff1, *Psix);                                // Psix = SinvRx|in>
    MatMatMult(H, *Psix, MAT_REUSE_MATRIX, PETSC_CURRENT, buff1); // buff1=HSinvRx|in>
    MatMatMult(Rx, *buff2, MAT_REUSE_MATRIX, PETSC_CURRENT, Psix); // Psix=Rx(SinvH|in>)
    MatAXPY(*buff1, -1.0, *Psix, SAME_NONZERO_PATTERN); // buff1=buff1-psix
    KSPMatSolve(ksp, *buff1, *Psix);

    // y
    MatMatMult(Ry, *Psi_in, MAT_REUSE_MATRIX, PETSC_CURRENT, buff1); 
    KSPMatSolve(ksp, *buff1, *Psiy);
    MatMatMult(H, *Psiy, MAT_REUSE_MATRIX, PETSC_CURRENT, buff1); // buff1=HSinvRy|in>
    MatMatMult(Ry, *buff2, MAT_REUSE_MATRIX, PETSC_CURRENT, Psiy); // Psix=Ry(SinvH|in>)
    MatAXPY(*buff1, -1.0, *Psiy, SAME_NONZERO_PATTERN); // buff1=buff1-Psix
    KSPMatSolve(ksp, *buff1, *Psiy);
    
    // z
    MatMatMult(Rz, *Psi_in, MAT_REUSE_MATRIX, PETSC_CURRENT, buff1); 
    KSPMatSolve(ksp, *buff1, *Psiz);
    MatMatMult(H, *Psiz, MAT_REUSE_MATRIX, PETSC_CURRENT, buff1); // buff1=HSinvRz|in>
    MatMatMult(Rz, *buff2, MAT_REUSE_MATRIX, PETSC_CURRENT, Psiz); // Psix=Rz(SinvH|in>)
    MatAXPY(*buff1, -1.0, *Psiz, SAME_NONZERO_PATTERN); // buff1=buff1-Psix
    KSPMatSolve(ksp, *buff1, *Psiz);

    const PetscComplex  nI(0.0, -1.0);

    MatScale(*Psix, nI);
    MatScale(*Psiy, nI);
    MatScale(*Psiz, nI);
}

void current_mult_adjoint(Mat H, KSP ksp, Mat Rx, Mat Ry, Mat Rz, Mat *Psi_in, Mat *Psix, Mat *Psiy, Mat *Psiz, Mat *buff1, Mat *buff2, Mat *buff3){
    KSPMatSolve(ksp, *Psi_in, *buff3);
    
    MatMatMult(H, *buff3, MAT_REUSE_MATRIX, PETSC_CURRENT, buff1);
    KSPMatSolve(ksp, *buff1, *buff2);

    // x
    MatMatMult(Rx, *buff3, MAT_REUSE_MATRIX, PETSC_CURRENT, Psix); 
    KSPMatSolve(ksp, *Psix, *buff1);
    MatMatMult(H, *buff1, MAT_REUSE_MATRIX, PETSC_CURRENT, Psix); // buff1=HSinvRx|in>
    MatMatMult(Rx, *buff2, MAT_REUSE_MATRIX, PETSC_CURRENT, buff1); // Psix=Rx(SinvH|in>)
    MatAXPY(*Psix, -1.0, *buff1, SAME_NONZERO_PATTERN); // buff1=buff1-psix

    // y
    MatMatMult(Ry, *buff3, MAT_REUSE_MATRIX, PETSC_CURRENT, Psiy); 
    KSPMatSolve(ksp, *Psiy, *buff1);
    MatMatMult(H, *buff1, MAT_REUSE_MATRIX, PETSC_CURRENT, Psiy); // buff1=HSinvRy|in>
    MatMatMult(Ry, *buff2, MAT_REUSE_MATRIX, PETSC_CURRENT, buff1); // Psix=Ry(SinvH|in>)
    MatAXPY(*Psiy, -1.0, *buff1, SAME_NONZERO_PATTERN); // buff1=buff1-Psix
    
    // z
    MatMatMult(Rz, *buff3, MAT_REUSE_MATRIX, PETSC_CURRENT, Psiz); 
    KSPMatSolve(ksp, *Psiz, *buff1);
    MatMatMult(H, *buff1, MAT_REUSE_MATRIX, PETSC_CURRENT, Psiz); // buff1=HSinvRz|in>
    MatMatMult(Rz, *buff2, MAT_REUSE_MATRIX, PETSC_CURRENT, buff1); // Psix=Rz(SinvH|in>)
    MatAXPY(*Psiz, -1.0, *buff1, SAME_NONZERO_PATTERN); // buff1=buff1-Psix

    const PetscComplex  nI(0.0, -1.0);

    MatScale(*Psix, nI);
    MatScale(*Psiy, nI);
    MatScale(*Psiz, nI);
}

template<typename T>
void corr_ac(Mat H, Mat Rx, Mat Ry, Mat Rz, KSP ksp, T Psi_in, int nt,  double temp, double mu, double scale, double shift, int n_kpm, double eps, PetscComplex *cac)
{
    double dt = M_PI;
    PetscComplex *coeff;
    PetscInt matdim, nrhs;
    if constexpr (std::is_same_v<T, Vec>){
        VecGetSize(Psi_in, &matdim);
    } else {
        MatGetSize(Psi_in, &matdim, &nrhs);
    }
    int n_bes=0;

    timeprop_coeff(dt, 1e-14, true, &coeff, &n_bes);

    if constexpr (std::is_same_v<T, Vec>){
        Vec Psi_x, Psi_y, Psi_z, Psi0, Psi0_x, Psi0_y, Psi0_z, *buff, Psi_out;

        //initialize these buffer
        VecDuplicateVecs(Psi_in, 3, &buff);
        VecDuplicate(Psi_in, &Psi_x);
        VecDuplicate(Psi_in, &Psi_y);
        VecDuplicate(Psi_in, &Psi_z);
        VecDuplicate(Psi_in, &Psi0);
        VecDuplicate(Psi_in, &Psi0_x);
        VecDuplicate(Psi_in, &Psi0_y);
        VecDuplicate(Psi_in, &Psi0_z);
        VecDuplicate(Psi_in, &Psi_out);

        // Psi0: apply fermi-dirac
        fermi_dirac_kpm<Vec>(H, ksp, Psi_in, Psi0, temp, mu, scale, shift, n_kpm, eps, true); // adjoint is open, since the Psi0 is left vector
    
        current(H, ksp, Rx, Ry, Rz, Psi_in, Psi_x, Psi_y, Psi_z, buff[0], buff[1]);

        fermi_dirac_kpm<Vec>(H, ksp, Psi_x, buff[1], temp, mu, scale, shift, n_kpm, eps);
        VecAXPY(Psi_x, -1.0, buff[1]);

        fermi_dirac_kpm<Vec>(H, ksp, Psi_y, buff[1], temp, mu, scale, shift, n_kpm, eps);
        VecAXPY(Psi_y, -1.0, buff[1]);
        
        fermi_dirac_kpm<Vec>(H, ksp, Psi_z, buff[1], temp, mu, scale, shift, n_kpm, eps);
        VecAXPY(Psi_z, -1.0, buff[1]);

        current_adjoint(H, ksp, Rx, Ry, Rz, Psi0, Psi0_x, Psi0_y, Psi0_z, buff[0], buff[1], buff[2]);
        
        VecDot(Psi0_x, Psi_x, cac); // xx, xy, xz, yy, yz, zz
        VecDot(Psi0_x, Psi_y, cac+1);
        VecDot(Psi0_x, Psi_z, cac+2);
        VecDot(Psi0_y, Psi_y, cac+3);
        VecDot(Psi0_y, Psi_z, cac+4);
        VecDot(Psi0_z, Psi_z, cac+5);

        for(int it=0; it<nt; ++it){
            std::cout << "corr_ac calculation: " << it << " / " << nt << "\n";
            // x
            cheby_recur(
                H, ksp, Psi_x,
                buff, // length 3 vec
                coeff, n_bes, 
                Psi_out
                );
            VecCopy(Psi_out, Psi_x);
            // y
            cheby_recur(
                H, ksp, Psi_y,
                buff, // length 3 vec
                coeff, n_bes, 
                Psi_out
                );
            VecCopy(Psi_out, Psi_y);
            // z
            cheby_recur(
                H, ksp, Psi_z,
                buff, // length 3 vec
                coeff, n_bes, 
                Psi_out
                );
            VecCopy(Psi_out, Psi_z);

            // Psi0
            cheby_recur(
                H, ksp, Psi0,
                buff, // length 3 vec
                coeff, n_bes, 
                Psi_out, true // adjoint is open, since the Psi0 is left vector
                );
            VecCopy(Psi_out, Psi0);
            
            current_adjoint(H, ksp, Rx, Ry, Rz, Psi0, Psi0_x, Psi0_y, Psi0_z, buff[0], buff[1], buff[2]);
            // ac_corr calc
            VecDot(Psi_x, Psi0_x, cac+(it+1)*6); // xx, xy, xz, yy, yz, zz
            VecDot(Psi_y, Psi0_x, cac+(it+1)*6+1);
            VecDot(Psi_z, Psi0_x, cac+(it+1)*6+2);
            VecDot(Psi_y, Psi0_y, cac+(it+1)*6+3);
            VecDot(Psi_z, Psi0_y, cac+(it+1)*6+4);
            VecDot(Psi_z, Psi0_z, cac+(it+1)*6+5);

        }

        VecDestroy(buff); VecDestroy(buff+1); VecDestroy(buff+2);
        VecDestroy(&Psi_x); VecDestroy(&Psi_y); VecDestroy(&Psi_z);
        VecDestroy(&Psi0_x); VecDestroy(&Psi0_y); VecDestroy(&Psi0_z);
        VecDestroy(&Psi0); VecDestroy(&Psi_out);
        
    } else {
        std::fill(cac, cac+(nt+1)*6, 0.0);

        Mat Psi_x, Psi_y, Psi_z, Psi0, Psi0_x, Psi0_y, Psi0_z, buff[4], Psi_out;
        Vec v_psi_left_x, v_psi_left_y, v_psi_left_z, v_psi_right_x, v_psi_right_y, v_psi_right_z;
        PetscScalar tmp;

        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &Psi_x);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &Psi_y);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &Psi_z);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &Psi0);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &Psi0_x);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &Psi0_y);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &Psi0_z);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &Psi_out);

        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff+1);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff+2);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff+3);
        // Psi0: apply fermi-dirac
        fermi_dirac_kpm<Mat>(H, ksp, Psi_in, Psi0, temp, mu, scale, shift, n_kpm, eps, true); // adjoint is open, since the Psi0 is left vector

        current_mult(H, ksp, Rx, Ry, Rz, &Psi_in, &Psi_x, &Psi_y, &Psi_z, buff, buff+1);

        fermi_dirac_kpm<Mat>(H, ksp, Psi_x, buff[1], temp, mu, scale, shift, n_kpm, eps);
        MatAXPY(Psi_x, -1.0, buff[1], SAME_NONZERO_PATTERN);

        fermi_dirac_kpm<Mat>(H, ksp, Psi_y, buff[1], temp, mu, scale, shift, n_kpm, eps);
        MatAXPY(Psi_y, -1.0, buff[1], SAME_NONZERO_PATTERN);

        fermi_dirac_kpm<Mat>(H, ksp, Psi_z, buff[1], temp, mu, scale, shift, n_kpm, eps);
        MatAXPY(Psi_z, -1.0, buff[1], SAME_NONZERO_PATTERN);

        current_mult_adjoint(H, ksp, Rx, Ry, Rz, &Psi0, &Psi0_x, &Psi0_y, &Psi0_z, buff, buff+1, buff+2);
        
        for(int in=0; in<nrhs; ++in){
            MatDenseGetColumnVecRead(Psi0_x, in, &v_psi_left_x);
            MatDenseGetColumnVecRead(Psi0_y, in, &v_psi_left_y);
            MatDenseGetColumnVecRead(Psi0_z, in, &v_psi_left_z);
            
            MatDenseGetColumnVecRead(Psi_x, in, &v_psi_right_x);
            MatDenseGetColumnVecRead(Psi_y, in, &v_psi_right_y);
            MatDenseGetColumnVecRead(Psi_z, in, &v_psi_right_z);

            VecDot(v_psi_right_x, v_psi_left_x, &tmp); // xx, xy, xz, yy, yz, zz
            cac[0] += tmp;
            VecDot(v_psi_right_y, v_psi_left_x, &tmp);
            cac[1] += tmp;
            VecDot(v_psi_right_z, v_psi_left_x, &tmp);
            cac[2] += tmp;
            VecDot(v_psi_right_y, v_psi_left_y, &tmp);
            cac[3] += tmp;
            VecDot(v_psi_right_z, v_psi_left_y, &tmp);
            cac[4] += tmp;
            VecDot(v_psi_right_z, v_psi_left_z, &tmp);
            cac[5] += tmp;

            MatDenseRestoreColumnVecRead(Psi_z, in, &v_psi_right_z);
            MatDenseRestoreColumnVecRead(Psi_y, in, &v_psi_right_y);
            MatDenseRestoreColumnVecRead(Psi_x, in, &v_psi_right_x);

            MatDenseRestoreColumnVecRead(Psi0_z, in, &v_psi_left_z);
            MatDenseRestoreColumnVecRead(Psi0_y, in, &v_psi_left_y);
            MatDenseRestoreColumnVecRead(Psi0_x, in, &v_psi_left_x);
        }
        cac[0] /= nrhs;
        cac[1] /= nrhs;
        cac[2] /= nrhs;
        cac[3] /= nrhs;
        cac[4] /= nrhs;
        cac[5] /= nrhs;

        for(int it=0; it<nt; ++it){
            std::cout << "corr_ac calculation: " << it << " / " << nt << "\n";
            // x
            cheby_recur_mult(
                H, ksp, Psi_x,
                buff, // length 3 vec
                coeff, n_bes, 
                Psi_out
                );
            MatCopy(Psi_out, Psi_x, SAME_NONZERO_PATTERN);
            // y
            cheby_recur_mult(
                H, ksp, Psi_y,
                buff, // length 3 vec
                coeff, n_bes, 
                Psi_out
                );
            MatCopy(Psi_out, Psi_y, SAME_NONZERO_PATTERN);
            // z
            cheby_recur_mult(
                H, ksp, Psi_z,
                buff, // length 3 vec
                coeff, n_bes, 
                Psi_out
                );
            MatCopy(Psi_out, Psi_z, SAME_NONZERO_PATTERN);

            // Psi0
            cheby_recur_mult(
                H, ksp, Psi0,
                buff, // length 3 vec
                coeff, n_bes, 
                Psi_out, true // adjoint is open, since the Psi0 is left vector
                );
            MatCopy(Psi_out, Psi0, SAME_NONZERO_PATTERN);
            
            current_mult_adjoint(H, ksp, Rx, Ry, Rz, &Psi0, &Psi0_x, &Psi0_y, &Psi0_z, buff, buff+1, buff+2);

            for(int in=0; in<nrhs; ++in){
                MatDenseGetColumnVecRead(Psi0_x, in, &v_psi_left_x);
                MatDenseGetColumnVecRead(Psi0_y, in, &v_psi_left_y);
                MatDenseGetColumnVecRead(Psi0_z, in, &v_psi_left_z);
                
                MatDenseGetColumnVecRead(Psi_x, in, &v_psi_right_x);
                MatDenseGetColumnVecRead(Psi_y, in, &v_psi_right_y);
                MatDenseGetColumnVecRead(Psi_z, in, &v_psi_right_z);

                VecDot(v_psi_right_x, v_psi_left_x, &tmp); // xx, xy, xz, yy, yz, zz
                cac[(it+1)*6] += tmp;
                VecDot(v_psi_right_y, v_psi_left_x, &tmp);
                cac[(it+1)*6+1] += tmp;
                VecDot(v_psi_right_z, v_psi_left_x, &tmp);
                cac[(it+1)*6+2] += tmp;
                VecDot(v_psi_right_y, v_psi_left_y, &tmp);
                cac[(it+1)*6+3] += tmp;
                VecDot(v_psi_right_z, v_psi_left_y, &tmp);
                cac[(it+1)*6+4] += tmp;
                VecDot(v_psi_right_z, v_psi_left_z, &tmp);
                cac[(it+1)*6+5] += tmp;

                MatDenseRestoreColumnVecRead(Psi_z, in, &v_psi_right_z);
                MatDenseRestoreColumnVecRead(Psi_y, in, &v_psi_right_y);
                MatDenseRestoreColumnVecRead(Psi_x, in, &v_psi_right_x);

                MatDenseRestoreColumnVecRead(Psi0_z, in, &v_psi_left_z);
                MatDenseRestoreColumnVecRead(Psi0_y, in, &v_psi_left_y);
                MatDenseRestoreColumnVecRead(Psi0_x, in, &v_psi_left_x);
            }

            cac[(it+1)*6] /= nrhs;
            cac[(it+1)*6+1] /= nrhs;
            cac[(it+1)*6+2] /= nrhs;
            cac[(it+1)*6+3] /= nrhs;
            cac[(it+1)*6+4] /= nrhs;
            cac[(it+1)*6+5] /= nrhs;
        }

        /* Destroy the buffer */
        MatDestroy(buff); MatDestroy(buff+1); MatDestroy(buff+2); MatDestroy(buff+3);
        MatDestroy(&Psi_x); MatDestroy(&Psi_y); MatDestroy(&Psi_z); MatDestroy(&Psi0_x); MatDestroy(&Psi0_y); MatDestroy(&Psi0_z);
        MatDestroy(&Psi0); MatDestroy(&Psi_out);

        VecDestroy(&v_psi_left_x); VecDestroy(&v_psi_left_y); VecDestroy(&v_psi_left_z);
        VecDestroy(&v_psi_right_x); VecDestroy(&v_psi_right_y); VecDestroy(&v_psi_right_z);
    }

}

template void corr_ac<Vec>(Mat, Mat, Mat, Mat, KSP, Vec, int,  double, double, double, double, int, double, PetscComplex*);
template void corr_ac<Mat>(Mat, Mat, Mat, Mat, KSP, Mat, int,  double, double, double, double, int, double, PetscComplex*);
}

#endif
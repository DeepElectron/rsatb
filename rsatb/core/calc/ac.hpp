#ifndef AC_CALC
#define AC_CALC

#include <petsc.h>
#include "cheby.hpp"
#include "fermi_dirac.hpp"
#include <math.h>

namespace rsatb::calc{

template<typename T>
void current(Mat H, KSP ksp, Mat Rx, Mat Ry, Mat Rz, T Psi_in, T *J_Psi, bool orthogonal=false){ 
    // apply iS^{-1}[H, R] on Psi_in
    // -i(e/\hbar)[\hat{H},\hat{R}] = iSinv(-HSinvR + RSinvH) -> -iSinv[R(-SinvHx) + HSinvRx] use atomic unit
    
    T buff1, buff2;
    if constexpr (std::is_same_v<T, Vec>){
        VecDuplicate(Psi_in, &buff1);
        VecDuplicate(Psi_in, &buff2);

        // buff2 = -S^{-1} H S^{-1} Psi_in
        MatMult(H, Psi_in, buff2);
        if(!orthogonal) KSPSolve(ksp, buff2, buff2);

        VecScale(buff2, -1.0);

        // x
        MatMult(Rx, Psi_in, buff1);
        if(!orthogonal) KSPSolve(ksp, buff1, buff1);
        MatMult(H, buff1, J_Psi[0]);
        MatMultAdd(Rx, buff2, J_Psi[0], J_Psi[0]);

        // y
        MatMult(Ry, Psi_in, buff1);
        if(!orthogonal) KSPSolve(ksp, buff1, buff1);
        MatMult(H, buff1, J_Psi[1]);
        MatMultAdd(Ry, buff2, J_Psi[1], J_Psi[1]);

        // z
        MatMult(Rz, Psi_in, buff1);
        if(!orthogonal) KSPSolve(ksp, buff1, buff1);
        MatMult(H, buff1, J_Psi[2]);
        MatMultAdd(Rz, buff2, J_Psi[2], J_Psi[2]);

        // S^{-1} solve
        if(!orthogonal){
            KSPSolve(ksp, J_Psi[0], J_Psi[0]);
            KSPSolve(ksp, J_Psi[1], J_Psi[1]);
            KSPSolve(ksp, J_Psi[2], J_Psi[2]);
        }

        const PetscComplex  nI(0.0, -1.0);
        VecScale(J_Psi[0], nI);
        VecScale(J_Psi[1], nI);
        VecScale(J_Psi[2], nI);

        VecDestroy(&buff1);
        VecDestroy(&buff2);

    } else {
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &buff1);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &buff2);

        MatMatMult(H, Psi_in, MAT_REUSE_MATRIX, PETSC_DEFAULT, &buff1);
        if(!orthogonal) KSPMatSolve(ksp, buff1, buff2);

        // x
        MatMatMult(Rx, Psi_in, MAT_REUSE_MATRIX, PETSC_DEFAULT, &buff1);
        if(!orthogonal) KSPMatSolve(ksp, buff1, J_Psi[0]);
        MatMatMult(H, J_Psi[0], MAT_REUSE_MATRIX, PETSC_DEFAULT, &buff1);
        MatMatMult(Rx, buff2, MAT_REUSE_MATRIX, PETSC_DEFAULT, J_Psi);
        MatAXPY(buff1, -1.0, J_Psi[0], SAME_NONZERO_PATTERN);
        if(!orthogonal) KSPMatSolve(ksp, buff1, J_Psi[0]);


        // y
        MatMatMult(Ry, Psi_in, MAT_REUSE_MATRIX, PETSC_DEFAULT, &buff1);
        if(!orthogonal) KSPMatSolve(ksp, buff1, J_Psi[1]);
        MatMatMult(H, J_Psi[1], MAT_REUSE_MATRIX, PETSC_DEFAULT, &buff1);
        MatMatMult(Ry, buff2, MAT_REUSE_MATRIX, PETSC_DEFAULT, J_Psi+1);
        MatAXPY(buff1, -1.0, J_Psi[1], SAME_NONZERO_PATTERN);
        if(!orthogonal) KSPMatSolve(ksp, buff1, J_Psi[1]);

        // z
        MatMatMult(Rz, Psi_in, MAT_REUSE_MATRIX, PETSC_DEFAULT, &buff1);
        if(!orthogonal) KSPMatSolve(ksp, buff1, J_Psi[2]);
        MatMatMult(H, J_Psi[2], MAT_REUSE_MATRIX, PETSC_DEFAULT, &buff1);
        MatMatMult(Rz, buff2, MAT_REUSE_MATRIX, PETSC_DEFAULT, J_Psi+2);
        MatAXPY(buff1, -1.0, J_Psi[2], SAME_NONZERO_PATTERN);
        if(!orthogonal) KSPMatSolve(ksp, buff1, J_Psi[2]);

        const PetscComplex  nI(0.0, -1.0);
        MatScale(J_Psi[0], nI);
        MatScale(J_Psi[1], nI);
        MatScale(J_Psi[2], nI);

        MatDestroy(&buff1);
        MatDestroy(&buff2);
    }
}

template<typename T>
void current_adjoint(Mat H, KSP ksp, Mat Rx, Mat Ry, Mat Rz, T Psi_in, T *J_Psi, bool orthogonal=false){ 
    // apply [R, H]S^{-1} on Psi_in
    // (-i[\hat{H},\hat{R}])* = i(RSinvH - HSinvR)Sinv -> -i[R(-SinvHSinvx) + HSinvRSinvx]
    
    T buff1, buff2, buff3;
    if constexpr (std::is_same_v<T, Vec>){
        VecDuplicate(Psi_in, &buff1);
        VecDuplicate(Psi_in, &buff2);
        VecDuplicate(Psi_in, &buff3);

        if(!orthogonal) KSPSolve(ksp, Psi_in, buff3);
        else VecCopy(Psi_in, buff3);

        MatMult(H, buff3, buff1);
        if(!orthogonal) KSPSolve(ksp, buff1, buff2); // buff2 = SinvHSinvx
        else VecCopy(buff1, buff2);
        VecScale(buff2, -1.0);

        // x
        MatMult(Rx, buff3, J_Psi[0]);
        if(!orthogonal) KSPSolve(ksp, J_Psi[0], buff1);
        else VecCopy(J_Psi[0], buff1);
        MatMult(H, buff1, J_Psi[0]);
        MatMultAdd(Rx, buff2, J_Psi[0], J_Psi[0]);

        // y
        MatMult(Ry, buff3, J_Psi[1]);
        if(!orthogonal) KSPSolve(ksp, J_Psi[1], buff1);
        else VecCopy(J_Psi[1], buff1);
        MatMult(H, buff1, J_Psi[1]);
        MatMultAdd(Ry, buff2, J_Psi[1], J_Psi[1]);

        // z
        MatMult(Rz, buff3, J_Psi[2]);
        if(!orthogonal) KSPSolve(ksp, J_Psi[2], buff1);
        else VecCopy(J_Psi[2], buff1);
        MatMult(H, buff1, J_Psi[2]);
        MatMultAdd(Rz, buff2, J_Psi[2], J_Psi[2]);

        const PetscComplex  nI(0.0, -1.0);
        VecScale(J_Psi[0], nI);
        VecScale(J_Psi[1], nI);
        VecScale(J_Psi[2], nI);

        VecDestroy(&buff1);
        VecDestroy(&buff2);
        VecDestroy(&buff3);
    } else {
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &buff1);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &buff2);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &buff3);

        if(!orthogonal) KSPMatSolve(ksp, Psi_in, buff3);
        else MatCopy(Psi_in, buff3, SAME_NONZERO_PATTERN);

        MatMatMult(H, buff3, MAT_REUSE_MATRIX, PETSC_DEFAULT, &buff1);
        if(!orthogonal) KSPMatSolve(ksp, buff1, buff2);
        else MatCopy(buff1, buff2, SAME_NONZERO_PATTERN);

        // x
        MatMatMult(Rx, buff3, MAT_REUSE_MATRIX, PETSC_DEFAULT, J_Psi);
        if(!orthogonal) KSPMatSolve(ksp, J_Psi[0], buff1);
        else MatCopy(J_Psi[0], buff1, SAME_NONZERO_PATTERN);
        MatMatMult(H, buff1, MAT_REUSE_MATRIX, PETSC_DEFAULT, J_Psi);
        MatMatMult(Rx, buff2, MAT_REUSE_MATRIX, PETSC_DEFAULT, &buff1);
        MatAXPY(J_Psi[0], -1.0, buff1, SAME_NONZERO_PATTERN);

        // y
        MatMatMult(Ry, buff3, MAT_REUSE_MATRIX, PETSC_DEFAULT, J_Psi+1);
        if(!orthogonal) KSPMatSolve(ksp, J_Psi[1], buff1);
        else MatCopy(J_Psi[1], buff1, SAME_NONZERO_PATTERN);
        MatMatMult(H, buff1, MAT_REUSE_MATRIX, PETSC_DEFAULT, J_Psi+1);
        MatMatMult(Ry, buff2, MAT_REUSE_MATRIX, PETSC_DEFAULT, &buff1);
        MatAXPY(J_Psi[1], -1.0, buff1, SAME_NONZERO_PATTERN);

        // z
        MatMatMult(Rz, buff3, MAT_REUSE_MATRIX, PETSC_DEFAULT, J_Psi+2);
        if(!orthogonal) KSPMatSolve(ksp, J_Psi[2], buff1);
        else MatCopy(J_Psi[2], buff1, SAME_NONZERO_PATTERN);
        MatMatMult(H, buff1, MAT_REUSE_MATRIX, PETSC_DEFAULT, J_Psi+2);
        MatMatMult(Rz, buff2, MAT_REUSE_MATRIX, PETSC_DEFAULT, &buff1);
        MatAXPY(J_Psi[2], -1.0, buff1, SAME_NONZERO_PATTERN);

        const PetscComplex  nI(0.0, -1.0);
        MatScale(J_Psi[0], nI);
        MatScale(J_Psi[1], nI);
        MatScale(J_Psi[2], nI);

        MatDestroy(&buff1);
        MatDestroy(&buff2);
        MatDestroy(&buff3);
    }
}

template<typename T>
void corr_ac(
    Mat H, Mat Rx, Mat Ry, Mat Rz, KSP ksp, T Psi_in, int nt, double temp, double mu, double scale, double shift, int n_kpm, 
    double eps=1e-14, PetscComplex *cac=nullptr, bool orthogonal=false, bool verbose=true){
    
    double dt = M_PI;
    PetscComplex *coeff;
    PetscInt matdim, nrhs;
    MPI_Comm comm;
    PetscObjectGetComm((PetscObject)H, &comm);
    int rank;
    MPI_Comm_rank(comm, &rank);

    if constexpr (std::is_same_v<T, Vec>){
        VecGetSize(Psi_in, &matdim);
        nrhs = 1;
    } else {
        MatGetSize(Psi_in, &matdim, &nrhs);
    }
    
    int n_bes=0;
    timeprop_coeff(dt, 1e-14, true, &coeff, &n_bes);

    std::fill(cac, cac+(nt+1)*9, 0.0);

    T J_Psi[3], Psi0, J_Psi0[3], Psi_out, buff[4];
    
    if constexpr (std::is_same_v<T, Vec>){
        VecDuplicate(Psi_in, &J_Psi[0]);
        VecDuplicate(Psi_in, &J_Psi[1]);
        VecDuplicate(Psi_in, &J_Psi[2]);
        VecDuplicate(Psi_in, &Psi0);
        VecDuplicate(Psi_in, &J_Psi0[0]);
        VecDuplicate(Psi_in, &J_Psi0[1]);
        VecDuplicate(Psi_in, &J_Psi0[2]);
        VecDuplicate(Psi_in, &Psi_out);
        VecDuplicate(Psi_in, &buff[0]);
        VecDuplicate(Psi_in, &buff[1]);
        VecDuplicate(Psi_in, &buff[2]);
    } else {
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &J_Psi[0]);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &J_Psi[1]);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &J_Psi[2]);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &Psi0);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &J_Psi0[0]);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &J_Psi0[1]);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &J_Psi0[2]);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &Psi_out);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &buff[0]);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &buff[1]);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &buff[2]);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, &buff[3]);
    }

    // Psi0: apply fermi-dirac (adjoint=true)
    fermi_dirac_kpm<T>(H, ksp, Psi_in, Psi0, temp, mu, scale, shift, n_kpm, eps, true, orthogonal);

    // J_Psi = current(Psi_in)
    current<T>(H, ksp, Rx, Ry, Rz, Psi_in, J_Psi, orthogonal);

    // Apply (1-f) to J_Psi
    // (1-f) J Psi = J Psi - f J Psi
    // Reuse buff[1] for temp
    if constexpr (std::is_same_v<T, Vec>){
        for(int i=0; i<3; ++i){
            fermi_dirac_kpm<T>(H, ksp, J_Psi[i], buff[1], temp, mu, scale, shift, n_kpm, eps, false, orthogonal);
            VecAXPY(J_Psi[i], -1.0, buff[1]);
        }
    } else {
        for(int i=0; i<3; ++i){
            fermi_dirac_kpm<T>(H, ksp, J_Psi[i], buff[1], temp, mu, scale, shift, n_kpm, eps, false, orthogonal);
            MatAXPY(J_Psi[i], -1.0, buff[1], SAME_NONZERO_PATTERN);
        }
    }

    // J_Psi0 = current_adjoint(Psi0)
    current_adjoint<T>(H, ksp, Rx, Ry, Rz, Psi0, J_Psi0, orthogonal);

    // Initial correlation
    PetscScalar tmp;
    for(int i=0; i<3; ++i){ // J_Psi index (right)
        for(int j=0; j<3; ++j){ // J_Psi0 index (left)
            if constexpr (std::is_same_v<T, Vec>){
                VecDot(J_Psi[i], J_Psi0[j], &tmp);
                cac[j*3+i] += tmp; // Store in cac[0..8]
            } else {
                Vec v_left, v_right;
                PetscScalar sum_val = 0.0;
                for(int in=0; in<nrhs; ++in){
                    MatDenseGetColumnVecRead(J_Psi0[j], in, &v_left);
                    MatDenseGetColumnVecRead(J_Psi[i], in, &v_right);
                    VecDot(v_right, v_left, &tmp);
                    sum_val += tmp;
                    MatDenseRestoreColumnVecRead(J_Psi0[j], in, &v_left);
                    MatDenseRestoreColumnVecRead(J_Psi[i], in, &v_right);
                }
                cac[j*3+i] += sum_val / nrhs;
            }
        }
    }

    // Time propagation
    for(int it=0; it<nt; ++it){
        if (rank == 0 && verbose) std::cout << "corr_ac calculation: " << it << " / " << nt << "\n";

        // Propagate J_Psi (right)
        for(int i=0; i<3; ++i){
            if constexpr (std::is_same_v<T, Vec>){
                cheby_recur(H, ksp, J_Psi[i], buff, coeff, n_bes, Psi_out, false, orthogonal);
                VecCopy(Psi_out, J_Psi[i]);
            } else {
                cheby_recur_mult(H, ksp, J_Psi[i], buff, coeff, n_bes, Psi_out, false, orthogonal);
                MatCopy(Psi_out, J_Psi[i], SAME_NONZERO_PATTERN);
            }
        }

        // Propagate Psi0 (left, adjoint)
        if constexpr (std::is_same_v<T, Vec>){
            cheby_recur(H, ksp, Psi0, buff, coeff, n_bes, Psi_out, true, orthogonal);
            VecCopy(Psi_out, Psi0);
        } else {
            cheby_recur_mult(H, ksp, Psi0, buff, coeff, n_bes, Psi_out, true, orthogonal);
            MatCopy(Psi_out, Psi0, SAME_NONZERO_PATTERN);
        }

        // Update J_Psi0 from propagated Psi0
        current_adjoint<T>(H, ksp, Rx, Ry, Rz, Psi0, J_Psi0, orthogonal);

        // Correlation
        for(int i=0; i<3; ++i){ // J_Psi index (right)
            for(int j=0; j<3; ++j){ // J_Psi0 index (left)
                if constexpr (std::is_same_v<T, Vec>){
                    VecDot(J_Psi[i], J_Psi0[j], &tmp);
                    cac[(it+1)*9 + j*3+i] += tmp;
                } else {
                    Vec v_left, v_right;
                    PetscScalar sum_val = 0.0;
                    for(int in=0; in<nrhs; ++in){
                        MatDenseGetColumnVecRead(J_Psi0[j], in, &v_left);
                        MatDenseGetColumnVecRead(J_Psi[i], in, &v_right);
                        VecDot(v_right, v_left, &tmp);
                        sum_val += tmp;
                        MatDenseRestoreColumnVecRead(J_Psi0[j], in, &v_left);
                        MatDenseRestoreColumnVecRead(J_Psi[i], in, &v_right);
                    }
                    cac[(it+1)*9 + j*3+i] += sum_val / nrhs;
                }
            }
        }
    }

    // Cleanup
    if constexpr (std::is_same_v<T, Vec>){
        VecDestroy(&J_Psi[0]); VecDestroy(&J_Psi[1]); VecDestroy(&J_Psi[2]);
        VecDestroy(&Psi0);
        VecDestroy(&J_Psi0[0]); VecDestroy(&J_Psi0[1]); VecDestroy(&J_Psi0[2]);
        VecDestroy(&Psi_out);
        VecDestroy(&buff[0]); VecDestroy(&buff[1]); VecDestroy(&buff[2]);
    } else {
        MatDestroy(&J_Psi[0]); MatDestroy(&J_Psi[1]); MatDestroy(&J_Psi[2]);
        MatDestroy(&Psi0);
        MatDestroy(&J_Psi0[0]); MatDestroy(&J_Psi0[1]); MatDestroy(&J_Psi0[2]);
        MatDestroy(&Psi_out);
        MatDestroy(&buff[0]); MatDestroy(&buff[1]); MatDestroy(&buff[2]); MatDestroy(&buff[3]);
    }
    delete coeff;
}

template void corr_ac<Vec>(Mat, Mat, Mat, Mat, KSP, Vec, int, double, double, double, double, int, double, PetscComplex*, bool, bool);
template void corr_ac<Mat>(Mat, Mat, Mat, Mat, KSP, Mat, int, double, double, double, double, int, double, PetscComplex*, bool, bool);

}

#endif
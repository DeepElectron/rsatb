#ifndef cheby
#define cheby

#include <petsc.h>
#include <mpi.h>
#include "bessel.hpp"


namespace rsatb::calc{

enum class ChebyMode {
    NORMAL,
    ADJOINT,
    CONJUGATE
};

void timeprop_coeff(const double& time_step, const double& precision, bool forward,
            /*output*/
            PetscComplex **coeff, int *n_bes){
    /* It is strictly the coeff of bessel function. 
    [J_0(z), 2(-i)^m J_m(z), ...]*/
    /* Always delete the coeff after using this function */
    
    double *bes_vec;
    const PetscComplex  I(0.0, 1.0);

    build_bessel_vector(time_step, precision, &bes_vec, n_bes);

    *coeff =  new PetscComplex[*n_bes];
    if(forward){
        for (int n = 0; n < *n_bes; ++n) {
            (*coeff)[n] = 2.0 * std::pow(-I, n) * bes_vec[n];
        }
    }else{
        for (int n = 0; n < *n_bes; ++n) {
            (*coeff)[n] = 2.0 * std::pow(I, n) * bes_vec[n];
        }
    }
    
    // First coefficient has a factor of 1 while others have 2.
    (*coeff)[0] *= 0.5;

    delete bes_vec;
}


/* The chebyshev recurance function

*/
inline void cheby_iter(Mat H, KSP ksp, Vec T_prev, Vec T_curr, Vec T_next, bool orthogonal=false){
    // T_next = - T_prev + 2 x M^(-1) x H x T_curr
    MatMult(H, T_curr, T_next);
    if(!orthogonal) KSPSolve(ksp, T_next, T_next);
    VecAXPBY(T_next, -1.0,  2.0, T_prev);

    VecSwap(T_prev, T_curr);
    VecSwap(T_curr, T_next);
}

inline void cheby_iter_mult(Mat H, KSP ksp, Mat *T_prev, Mat *T_curr, Mat *T_next, Mat *T_tmp, bool orthogonal=false){
    MatMatMult(H, *T_curr, MAT_REUSE_MATRIX, PETSC_DEFAULT, T_tmp);
    if(!orthogonal) {
        KSPMatSolve(ksp, *T_tmp, *T_next);
    } else {
        MatCopy(*T_tmp, *T_next, SAME_NONZERO_PATTERN);
    }
    MatScale(*T_next, 2.0);
    MatAXPY(*T_next, -1.0, *T_prev, SAME_NONZERO_PATTERN);

    Mat tmp=*T_prev;
    *T_prev = *T_curr; *T_curr = tmp;
    *T_curr = *T_next; *T_next = tmp;
    // MatSwap(*T_prev, *T_curr);
    // MatSwap(*T_curr, *T_next);
}

inline void cheby_iter_adjoint(Mat H, KSP ksp, Vec T_prev, Vec T_curr, Vec T_next, bool orthogonal=false){
    // T_next = - T_prev + 2 x H x M^(-1) x T_curr
    if(!orthogonal) {
        KSPSolve(ksp, T_curr, T_next);
    } else {
        VecCopy(T_curr, T_next);
    }
    VecScale(T_next, 2.0);
    VecScale(T_prev, -1.0);
    MatMultAdd(H, T_next, T_prev, T_prev);

    VecSwap(T_prev, T_curr);
}

inline void cheby_iter_adjoint_mult(Mat H, KSP ksp, Mat *T_prev, Mat *T_curr, Mat *T_next, Mat *T_tmp, bool orthogonal=false){
    if(!orthogonal) {
        KSPMatSolve(ksp, *T_curr, *T_tmp);
    } else {
        MatCopy(*T_curr, *T_tmp, SAME_NONZERO_PATTERN);
    }
    MatMatMult(H, *T_tmp, MAT_REUSE_MATRIX, PETSC_DEFAULT, T_next);
    MatScale(*T_next, 2.0);
    MatAXPY(*T_next, -1.0, *T_prev, SAME_NONZERO_PATTERN);

    Mat tmp=*T_prev;
    *T_prev = *T_curr; *T_curr = tmp;
    *T_curr = *T_next; *T_next = tmp;
    // MatSwap(*T_prev, *T_curr);
    // MatSwap(*T_curr, *T_next);
}

inline void cheby_iter_conjugate(Mat H, KSP ksp, Vec T_prev, Vec T_curr, Vec T_next, bool orthogonal=false){
    // T_next = - T_prev + 2 x (S^-1)^T x H^T x T_curr
    MatMultTranspose(H, T_curr, T_next);
    if(!orthogonal) KSPSolveTranspose(ksp, T_next, T_next);
    VecAXPBY(T_next, -1.0,  2.0, T_prev);

    VecSwap(T_prev, T_curr);
    VecSwap(T_curr, T_next);
}

inline void cheby_iter_conjugate_mult(Mat H, KSP ksp, Mat *T_prev, Mat *T_curr, Mat *T_next, Mat *T_tmp, bool orthogonal=false){
    MatTransposeMatMult(H, *T_curr, MAT_REUSE_MATRIX, PETSC_DEFAULT, T_tmp);
    if(!orthogonal) {
        KSPMatSolveTranspose(ksp, *T_tmp, *T_next);
    } else {
        MatCopy(*T_tmp, *T_next, SAME_NONZERO_PATTERN);
    }
    MatScale(*T_next, 2.0);
    MatAXPY(*T_next, -1.0, *T_prev, SAME_NONZERO_PATTERN);

    Mat tmp=*T_prev;
    *T_prev = *T_curr; *T_curr = tmp;
    *T_curr = *T_next; *T_next = tmp;
}

inline void cheby_recur(
    Mat H, KSP ksp, Vec Psi_in,
    Vec *buff, // length 3 vec
    PetscComplex *coeff, PetscInt n_coeff, 
    Vec Psi_out, ChebyMode mode=ChebyMode::NORMAL, bool orthogonal=false
    ){
    if(n_coeff < 3)
        throw std::invalid_argument("chebyshev_propagate: need at least three Bessel coefficients");

    // const PetscComplex  I(0.0, 1.0);
    // — scratch buffers if none —
    if(buff==nullptr){
        VecDuplicateVecs(Psi_in, 3, &buff);
    }

    
    if( mode == ChebyMode::ADJOINT ){
        /* init T_prev(buff[0])
        HS^(-1)|Psi>    */
        if(!orthogonal) {
            KSPSolve(ksp, Psi_in, buff[1]);
        } else {
            VecCopy(Psi_in, buff[1]);
        }
        MatMult(H, buff[1], buff[0]);

         /* init T_curr(buff[1])
            2HS^(-1)|buff[0]> - |Psi>
        */
        
        if(!orthogonal) {
            KSPSolve(ksp, buff[0], buff[2]);
        } else {
            VecCopy(buff[0], buff[2]);
        }
        MatMult(H, buff[2], buff[1]);
        VecAXPBY(buff[1], -1.0, 2.0, Psi_in);
    } else if ( mode == ChebyMode::CONJUGATE ) {
        /* init T_prev(buff[0])
        (S^-1)^T H^T |Psi>    */
        MatMultTranspose(H, Psi_in, buff[0]);
        if(!orthogonal) KSPSolveTranspose(ksp, buff[0], buff[0]);

        /* init T_curr(buff[1])
            2(S^-1)^T H^T |buff[0]> - |Psi>
        */
        MatMultTranspose(H, buff[0], buff[1]);
        if(!orthogonal) KSPSolveTranspose(ksp, buff[1], buff[1]);
        VecAXPBY(buff[1], -1.0, 2.0, Psi_in);
    } else {
        /* init T_prev(buff[0])
        S^(-1)H|Psi>    */
        MatMult(H, Psi_in, buff[0]);
        // std::cout<<"Here?!\n";
        if(!orthogonal) KSPSolve(ksp, buff[0], buff[0]);
        // VecScale(buff[0], alpha1);

        /* init T_curr(buff[1])
            2S^(-1)H|buff[0]> - |Psi>
        */
        MatMult(H, buff[0], buff[1]);
        if(!orthogonal) KSPSolve(ksp, buff[1], buff[1]);
        VecAXPBY(buff[1], -1.0, 2.0, Psi_in);
    }

    // compute current scalar output
    VecCopy(Psi_in,Psi_out);
    VecAXPBYPCZ(Psi_out, coeff[1], coeff[2], coeff[0], buff[0], buff[1]);

    // Higher-order Chebyshev recurrence
    if ( mode == ChebyMode::ADJOINT ){
        for(std::size_t m = 3; m < n_coeff; ++m){
            cheby_iter_adjoint(H, ksp, buff[0], buff[1], buff[2], orthogonal);
            VecAXPBY(Psi_out, coeff[m], 1.0, buff[1]);
        }
    } else if ( mode == ChebyMode::CONJUGATE ) {
        for(std::size_t m = 3; m < n_coeff; ++m){
            cheby_iter_conjugate(H, ksp, buff[0], buff[1], buff[2], orthogonal);
            VecAXPBY(Psi_out, coeff[m], 1.0, buff[1]);
        }
    } else {
        for(std::size_t m = 3; m < n_coeff; ++m){
            cheby_iter(H, ksp, buff[0], buff[1], buff[2], orthogonal);
            VecAXPBY(Psi_out, coeff[m], 1.0, buff[1]);
        }
    }

}

inline void cheby_recur_mult(
    Mat H, KSP ksp, Mat Psi_in,
    Mat *buff, // length 4 Mat buff
    PetscComplex *coeff, PetscInt n_coeff, 
    Mat Psi_out, ChebyMode mode=ChebyMode::NORMAL, bool orthogonal=false
    ){
    if(n_coeff < 3)
        throw std::invalid_argument("chebyshev_propagate: need at least three Bessel coefficients");

    // const PetscComplex  I(0.0, 1.0);
    // — scratch buffers —
    if(buff==nullptr){
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff+1);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff+2);
        MatDuplicate(Psi_in, MAT_DO_NOT_COPY_VALUES, buff+3);
    }
    
    if ( mode == ChebyMode::ADJOINT ){
        // init T_prev(buff[0])
        if(!orthogonal) {
            KSPMatSolve(ksp, Psi_in, buff[3]);
        } else {
            MatCopy(Psi_in, buff[3], SAME_NONZERO_PATTERN);
        }
        MatMatMult(H, buff[3], MAT_REUSE_MATRIX, PETSC_DEFAULT, buff);

        // init T_curr(buff[1])
        if(!orthogonal) {
            KSPMatSolve(ksp, buff[0], buff[3]);
        } else {
            MatCopy(buff[0], buff[3], SAME_NONZERO_PATTERN);
        }
        MatMatMult(H, buff[3], MAT_REUSE_MATRIX, PETSC_DEFAULT, buff+1);
        MatScale(buff[1], 2.0);
        MatAXPY(buff[1], -1.0, Psi_in, SAME_NONZERO_PATTERN);
    } else if ( mode == ChebyMode::CONJUGATE ) {
        // init T_prev(buff[0])
        MatTransposeMatMult(H, Psi_in, MAT_REUSE_MATRIX, PETSC_DEFAULT, buff+3);
        if(!orthogonal) {
            KSPMatSolveTranspose(ksp, buff[3], buff[0]);
        } else {
            MatCopy(buff[3], buff[0], SAME_NONZERO_PATTERN);
        }

        // init T_curr(buff[1])
        MatTransposeMatMult(H, buff[0], MAT_REUSE_MATRIX, PETSC_DEFAULT, buff+3);
        if(!orthogonal) {
            KSPMatSolveTranspose(ksp, buff[3], buff[1]);
        } else {
            MatCopy(buff[3], buff[1], SAME_NONZERO_PATTERN);
        }
        MatScale(buff[1], 2.0);
        MatAXPY(buff[1], -1.0, Psi_in, SAME_NONZERO_PATTERN);
    } else {
        // init T_prev(buff[0])
        MatMatMult(H, Psi_in, MAT_REUSE_MATRIX, PETSC_DEFAULT, buff+3);
        if(!orthogonal) {
            KSPMatSolve(ksp, buff[3], buff[0]);
        } else {
            MatCopy(buff[3], buff[0], SAME_NONZERO_PATTERN);
        }

        // init T_curr(buff[1])
        MatMatMult(H, buff[0], MAT_REUSE_MATRIX, PETSC_DEFAULT, buff+3);
        if(!orthogonal) {
            KSPMatSolve(ksp, buff[3], buff[1]);
        } else {
            MatCopy(buff[3], buff[1], SAME_NONZERO_PATTERN);
        }
        MatScale(buff[1], 2.0);
        MatAXPY(buff[1], -1.0, Psi_in, SAME_NONZERO_PATTERN);
    }

    // compute current scalar output
    MatCopy(Psi_in, Psi_out, SAME_NONZERO_PATTERN);
    MatScale(Psi_out, coeff[0]);
    MatAXPY (Psi_out, coeff[1],buff[0],SAME_NONZERO_PATTERN);
    MatAXPY (Psi_out, coeff[2],buff[1],SAME_NONZERO_PATTERN);


    // Higher-order Chebyshev recurrence
    if ( mode == ChebyMode::ADJOINT ){
        for(std::size_t m = 3; m < n_coeff; ++m){
            cheby_iter_adjoint_mult(H, ksp, buff, buff+1, buff+2, buff+3, orthogonal);
            MatAXPY(Psi_out, coeff[m], buff[1], SAME_NONZERO_PATTERN);
        }
    } else if ( mode == ChebyMode::CONJUGATE ) {
        for(std::size_t m = 3; m < n_coeff; ++m){
            cheby_iter_conjugate_mult(H, ksp, buff, buff+1, buff+2, buff+3, orthogonal);
            MatAXPY(Psi_out, coeff[m], buff[1], SAME_NONZERO_PATTERN);
        }
    } else {
        for(std::size_t m = 3; m < n_coeff; ++m){
            cheby_iter_mult(H, ksp, buff, buff+1, buff+2, buff+3, orthogonal);
            MatAXPY(Psi_out, coeff[m], buff[1], SAME_NONZERO_PATTERN);
        }
    }

}

};


#endif
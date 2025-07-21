#ifndef cheby
#define cheby

#include <petsc.h>
#include <mpi.h>


namespace rsatb::calc{


void timeprop_coeff(const double& time_step, const double& precision, bool forward,
            /*output*/
            PetscComplex **coeff, int *n_bes){
    /* It is strictly the coeff of bessel function. 
    [J_0(z), 2(-i)^m J_m(z), ...]*/
    double *bes_vec;
    const PetscComplex  I(0.0, 1.0);

    build_bessel_vector(time_step, bessel_precision, &bes_vec, &n_bes);

    size_t num_bes = bes_vec.size();
    PetscComplex coeff_[n_bes];
    if(forward){
        for (int n = 0; n < n_bes; ++n) {
            coeff_[n] = 2.0 * std::pow(-I, n) * bes_vec[n];
        }
    }else{
        coeff_[n] = 2.0 * std::pow(I, n) * bes_vec[n];
    }
    
    // First coefficient has a factor of 1 while others have 2.
    coeff_[0] *= 0.5;
    coeff_[0] *= 0.5;

    *coeff = coeff_;
}


/* The chebyshev recurance function

*/
inline void cheby_iter(Mat H, KSP ksp, Vec T_prev, Vec T_curr, Vec T_next){
    // T_next = - T_prev + 2 x M^(-1) x H x T_curr
    MatMult(H, T_curr, T_next);
    KSPSolve(ksp, T_next, T_next);
    VecAXPBY(T_next, -1.0,  2.0, T_prev);

    VecSwap(T_prev, T_curr);
    VecSwap(T_curr, T_next);
}

inline void cheby_iter_mult(Mat H, KSP ksp, Mat *T_prev, Mat *T_curr, Mat *T_next, Mat *T_tmp){
    MatMatMult(H, *T_curr, MAT_REUSE_MATRIX, PETSC_CURRENT, T_tmp);
    KSPMatSolve(ksp, *T_tmp, *T_next);
    MatScale(*T_next, 2.0);
    MatAXPY(*T_next, -1.0, *T_prev, SAME_NONZERO_PATTERN);

    Mat tmp=*T_prev;
    *T_prev = *T_curr; *T_curr = tmp;
    *T_curr = *T_next; *T_next = tmp;
    // MatSwap(*T_prev, *T_curr);
    // MatSwap(*T_curr, *T_next);
}


inline void cheby_recur(
    Mat H, KSP ksp, Vec Psi_in,
    Vec *buff, // length 3 vec
    PetscReal *coeff, PetscInt n_coeff, 
    Vec Psi_out
    ){
    if(n_coeff < 3)
        throw std::invalid_argument("chebyshev_propagate: need at least three Bessel coefficients");

    // const PetscComplex  I(0.0, 1.0);
    // — scratch buffers if none —
    if(buff==nullptr){
        VecDuplicateVecs(Psi_in, 3, &buff);
    }

    /* init T_prev(buff[0])
        S^(-1)H|Psi>    */
    MatMult(H, Psi_in, buff[0]);
    // std::cout<<"Here?!\n";
    KSPSolve(ksp, buff[0], buff[0]);
    // VecScale(buff[0], alpha1);

    /* init T_curr(buff[1])
        2S^(-1)H|buff[0]> - |Psi>
    */
    MatMult(H, buff[0], buff[1]);
    KSPSolve(ksp, buff[1], buff[1]);
    VecAXPBY(buff[1], -1.0, 2.0, Psi_in);

    // compute current scalar output
    VecCopy(Psi_in,Psi_out);
    VecAXPBYPCZ(Psi_out, (PetscScalar)coeff[1], (PetscScalar)coeff[2], (PetscScalar)coeff[0], buff[0], buff[1]);

    // Higher-order Chebyshev recurrence
    
    for(std::size_t m = 3; m < n_coeff; ++m){
        cheby_iter(H, ksp, buff[0], buff[1], buff[2]);
        VecAXPBY(Psi_out, (PetscScalar)coeff[m], 1.0, buff[1]);
    }
}

inline void cheby_recur_mult(
    Mat H, KSP ksp, Mat Psi_in,
    Mat *buff, // length 4 Mat buff
    PetscReal *coeff, PetscInt n_coeff, 
    Mat Psi_out
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
    

    // init T_prev(buff[0])
    MatMatMult(H, Psi_in, MAT_REUSE_MATRIX, PETSC_CURRENT, buff+3);
    KSPMatSolve(ksp, buff[3], buff[0]);

    // init T_curr(buff[1])
    MatMatMult(H, buff[0], MAT_REUSE_MATRIX, PETSC_CURRENT, buff+3);
    KSPMatSolve(ksp, buff[3], buff[1]);
    MatScale(buff[1], 2.0);
    MatAXPY(buff[1], -1.0, Psi_in, SAME_NONZERO_PATTERN);

    // compute current scalar output
    MatCopy(Psi_in, Psi_out, SAME_NONZERO_PATTERN);
    MatScale(Psi_out,(PetscScalar)coeff[0]);
    MatAXPY (Psi_out,(PetscScalar)coeff[1],buff[0],SAME_NONZERO_PATTERN);
    MatAXPY (Psi_out,(PetscScalar)coeff[2],buff[1],SAME_NONZERO_PATTERN);

    // Higher-order Chebyshev recurrence
    for(std::size_t m = 3; m < n_coeff; ++m){
        cheby_iter_mult(H, ksp, buff, buff+1, buff+2, buff+3);
        MatAXPY(Psi_out, (PetscScalar)coeff[m], buff[1], SAME_NONZERO_PATTERN);
    }
}

};


#endif
#ifndef ISQRTM_CALC
#define ISQRTM_CALC
// matrix inverse square root calculation using dbcsr
#include <dbcsr.h>
#include <backend/dbcsr.hpp>
#include <backend/dbcsr_capi.h>

namespace rsatb::calc{

void isqrtm(Mat mat, DM dm, double tol=1e-7, double sptol=1e-6, int maxit = 20, double factor=0.31, Mat *mat_isqrt=nullptr){
    if (*mat_isqrt==nullptr) {
        DMCreateMatrix(dm, mat_isqrt);
        MatAssemblyBegin(*mat_isqrt, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(*mat_isqrt, MAT_FINAL_ASSEMBLY);
    }

    MatZeroEntries(*mat_isqrt);
    MatShift(*mat_isqrt, factor);

    dbcsr_matrix Z, S, tmp1, tmp2, I;
    dbcsr_distribution dist;
    long long int cflop=0;
    int count=0, rank;
    double norm, norm_base, norm_pre=0;
    PetscInt vStart, vEnd;
    int nLocal, NGlobal;
    PetscBool ghost;

    // check_density(mat);
    double error = 999.0;

    // https://github.com/cp2k/cp2k/blob/e68414f03635ab23d4f4fbfa5fd8081a7e533449/src/dm_ls_scf_methods.F#L60
    // https://github.com/cp2k/cp2k/blob/e68414f03635ab23d4f4fbfa5fd8081a7e533449/src/iterate_matrix.F#L62
    // The implementation of CP2K about the purification method for DM can be a good reference
    // learn how to use DBCSR to efficiently compute matrix multiplication and purification based calculation

    MPI_Comm comm;
    PetscObjectGetComm((PetscObject)dm, &comm);
    MPI_Comm_rank(comm, &rank);

    DMNetworkGetVertexRange(dm, &vStart, &vEnd);
    nLocal = 0;
    for(int k=vStart; k<vEnd; ++k){
        DMNetworkIsGhostVertex(dm,k,&ghost);
        if(!ghost){
            nLocal++;
        }
    }

    NGlobal = nLocal;
    MPI_Allreduce(MPI_IN_PLACE, &NGlobal, 1, MPI_INT, MPIU_SUM, comm);

    DMat2DBCSR(mat, dm, &S, 'H');
    DMat2DBCSR(*mat_isqrt, dm, &Z, 'N');
    c_dbcsr_get_distribution(S, &dist);

    // double _Complex tr_d = mk_c99(0.0, 0.0);
    // c_dbcsr_trace_z(S, &tr_d);

    // std::cout << std::complex<double>(tr_d) << std::endl;

    // c_dbcsr_trace_z(Z, &tr_d);

    // std::cout << std::complex<double>(tr_d) << std::endl;

    std::vector<int> blksize(NGlobal);
    c_dbcsr_get_col_blk_size(S, blksize.data(), NGlobal);

    // c_dbcsr_create_template(&Z, "MatISQRT", S, dist, "N", nullptr, NGlobal, nullptr, NGlobal, nullptr, nullptr, nullptr, nullptr, nullptr);
    c_dbcsr_create_new(&tmp1, "tmp1", dist, 'N', blksize.data(), NGlobal, blksize.data(), NGlobal, nullptr, &dbcsr_type_complex_8, nullptr, nullptr, nullptr, nullptr);
    c_dbcsr_finalize(tmp1);
    c_dbcsr_create_new(&tmp2, "tmp2", dist, 'N', blksize.data(), NGlobal, blksize.data(), NGlobal, nullptr, &dbcsr_type_complex_8, nullptr, nullptr, nullptr, nullptr);
    c_dbcsr_finalize(tmp2);



    // create I with complex*16 type
    c_dbcsr_create_new(&I, "I", dist, 'N', blksize.data(), NGlobal, blksize.data(), NGlobal, nullptr, &dbcsr_type_complex_8, nullptr, nullptr, nullptr, nullptr);
    c_dbcsr_reserve_diag_blocks(I);
    // fill each diagonal block with an identity (1 on sub-diagonal)
    bool sum = false;                                // overwrite
    int bs;
    std::complex<double> scale=1.0;
    for (int bi = 0; bi < NGlobal; ++bi) {
        bs = blksize[bi];                       // block size
        std::vector<std::complex<double>> blk((size_t)bs*bs, 0.0);
        for (int k=0; k<bs; ++k) blk[(size_t)k*bs + k] = 1.0;

        c_dbcsr_put_block2d_z_api(I, bi, bi, blk.data(), bs, bs, &sum, &scale);
    }
    c_dbcsr_finalize(I);

    // double _Complex tr_d = mk_c99(0.0, 0.0);

    // std::cout << "Here?" << std::endl;

    c_dbcsr_multiply_z('C', 'N', 1.0, Z, S, 0.0, tmp1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &sptol, &cflop);
    // std::cout << "Here?" << std::endl;
    c_dbcsr_multiply_z('N', 'N', 1.0, tmp1, Z, 0.0, tmp2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &sptol, &cflop);
    // std::cout << "Here?" << std::endl;
    c_dbcsr_add_z(tmp2, I, 1.0, -1.0);
    bool local=false;
    // std::cout << "Here?" << std::endl;
    norm_base = c_dbcsr_frobenius_norm(tmp2, &local);
    c_dbcsr_add_z(tmp2, I, 1.0, 1.0);
    // std::cout << "Here?" << std::endl;
    norm = norm_base;

    // std::cout << "Here?" << std::endl;

    // check whether the convert is correct
    // bool ll=false;
    // double c_norm;
    // PetscReal nrm;
    // Mat newm;
    // c_dbcsr_multiply_z('N', 'N', 1.0, S, S, 0.0, tmp1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &sptol, &cflop);
    // MatMatMult(mat, mat, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &newm);
    // c_norm = c_dbcsr_frobenius_norm(tmp1, &ll);
    // MatNorm(newm, NORM_FROBENIUS, &nrm);
    // std::cout << c_norm << " " << nrm << std::endl;

    while (norm/norm_base > tol && count < maxit && std::abs(norm_pre-norm) / norm_base > 0.1*tol)
    {
        if (rank == 0){
            std::cout << "Recursive Factorization of S: current error: F norm - " << norm << " relative F norm - " << norm / norm_base  << std::endl;
        }

        norm_pre = norm;

        c_dbcsr_multiply_z('N', 'N', 1.0, tmp2, tmp2, 0.0, tmp1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &sptol, &cflop);
        
        // update Z
        c_dbcsr_add_z(tmp1, tmp2, 0.375, -1.25);
        c_dbcsr_add_z(tmp1, I, 1.0, 1.875);
        
        c_dbcsr_filter(tmp1, &sptol, nullptr, nullptr, nullptr);
        c_dbcsr_multiply_z('N', 'N', 1.0, Z, tmp1, 0.0, tmp2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &sptol, &cflop);
        // c_dbcsr_copy(&Z, tmp2, nullptr, nullptr, nullptr, nullptr, nullptr);
        c_dbcsr_add_z(Z, tmp2, 0.0, 1.0);
        

        // double _Complex tr_d = mk_c99(0.0, 0.0);
        // c_dbcsr_trace_z(Z, &tr_d);

        // std::cout << std::complex<double>(tr_d) << std::endl;

        // this->check_density(_Sr_);

        c_dbcsr_multiply_z('C', 'N', 1.0, Z, S, 0.0, tmp1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &sptol, &cflop);
        c_dbcsr_multiply_z('N', 'N', 1.0, tmp1, Z, 0.0, tmp2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &sptol, &cflop);

        c_dbcsr_add_z(tmp2, I, 1.0, -1.0);
        norm = c_dbcsr_frobenius_norm(tmp2, nullptr);
        c_dbcsr_add_z(tmp2, I, 1.0,  1.0);
        
        count ++;
    }

    std::cout << "Recursive Factorization of S: Converge at F norm - " << norm << " relative F norm - " << norm / norm_base << std::endl;
    
    DBCSR2Mat(dm, Z, mat_isqrt);
    c_dbcsr_release(&S);
    c_dbcsr_release(&tmp1);
    c_dbcsr_release(&tmp2);
    c_dbcsr_release(&I);
    c_dbcsr_distribution_release(&dist);
};

// void check_density(Mat mat){
//     PetscInt  M,N;
//     MatInfo   mi;
//     MatGetSize(mat,&M,&N);                         // global rows/cols
//     MatGetInfo(mat, MAT_GLOBAL_SUM, &mi);          // after assembly, collective when MAT_GLOBAL_SUM
//     PetscReal density = (M>0 && N>0)
//             ? (PetscReal)mi.nz_used / ((PetscReal)M*(PetscReal)N)
//             : 0.0;
//     if (rank == 0){
//         std::cout << "Mat Summary: nnz - " << mi.nz_used << " with total allocation - " << mi.nz_allocated << " density - " << density << std::endl;
//     }
// }

}

#endif
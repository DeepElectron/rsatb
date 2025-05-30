#ifndef CALC
#define CALC
#include <complex>
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#include <mkl.h>
#include <boost/math/special_functions/bessel.hpp>
#include "propagation_cg.hpp"
#include "propagation.hpp"
#include "propagation_direct.hpp"
#include <optional>
#include <complex>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cstddef>
#include <mpi.h>

inline std::vector<cd> tdos(
                int* ih, 
                int* jh, 
                cd* h,
                cd* Psi_in,
                std::size_t n,
                std::size_t nrhs,
                std::vector<double>& ts,
                int* is=nullptr,
                int* js=nullptr,
                cd* s=nullptr,
                bool forward=true,
                double btol=1e-20
        ){
    
    int Bmax = 100;
    bool isoverlap = is && js && s;

    // create Psi_t and initialize
    std::vector<cd> Psi_t(n * nrhs);
    int stride = 1;
    int nnrhs = n * nrhs;

    std::cout << "[DEBUG] copy the Psi_t " << std::endl;

    zcopy(&nnrhs, reinterpret_cast<MKL_Complex16*>(Psi_in), &stride, reinterpret_cast<MKL_Complex16*>(Psi_t.data()), &stride);


    std::cout << "[DEBUG] copy end for the Psi_t " << std::endl;
    
    // create MKL H handler
    sparse_matrix_t H = nullptr;
    sparse_status_t status;
    matrix_descr descr;

    descr.type = SPARSE_MATRIX_TYPE_HERMITIAN;
    descr.mode = SPARSE_FILL_MODE_UPPER;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    std::cout << "[DEBUG] create MKL handler " << std::endl;

    status = mkl_sparse_z_create_csr(&H, SPARSE_INDEX_BASE_ZERO,
                            n, n, ih, ih + 1, jh,
                            reinterpret_cast<MKL_Complex16*>(h));
    if (status != SPARSE_STATUS_SUCCESS || !H)
        throw std::runtime_error("[ERROR] MKL failed to create H csr matrix!");

    // create overlap handler if needed
    sparse_matrix_t S = nullptr;
    matrix_descr ovpdescr;
    ovpdescr.type = SPARSE_MATRIX_TYPE_HERMITIAN;
    ovpdescr.mode = SPARSE_FILL_MODE_UPPER;
    ovpdescr.diag = SPARSE_DIAG_UNIT;

    void *pt[64] = { nullptr };
    MKL_INT iparm[64];
    MKL_INT maxfct=1, mnum=1, phase=33, error=0, msglvl=1, mtype=4; 
    MKL_INT *perm=nullptr;
    MKL_INT N    = static_cast<MKL_INT>(n);
    MKL_INT NRHS = static_cast<MKL_INT>(nrhs);
    cd ddum;

    HPDDM::MatrixCSR<cd> S_;
    std::unique_ptr<OverlapOperator> op;

    std::cout << "[DEBUG] construction of overlap Operator " << std::endl;

    if (isoverlap){
         // create the overlap MKL handler;
        status = mkl_sparse_z_create_csr(&S, SPARSE_INDEX_BASE_ZERO,
                        n, n, is, is + 1, js,
                        reinterpret_cast<MKL_Complex16*>(s));
        if (status != SPARSE_STATUS_SUCCESS || !S)
        throw std::runtime_error("[ERROR] MKL failed to create S csr matrix!");

        if (n < 1e0) { // use direct solver
            // initialize the solver param
            std::cout << "[DEBUG] perform Pardiso initialization " << std::endl;

            for (int i=1; i<64;i++){
                iparm[i] = 0;
            }
            iparm[0] = 64;
            iparm[1] = 2; // use the METIS sorting algorithm
            iparm[3] = 0; // preconditioner
            iparm[5] = 0; // write solution on x, not for this stage
            iparm[7] = 2;
            iparm[9] = 13; // 
            iparm[23] = 1; // 0 for classical, 1 for factorize large matrix
            iparm[24] = 0; // parralization for solving multiple RHS, 0 for that across RHS, 1 for serial, 2 for that across matrix
            iparm[34] = 1;

            // start solving
            // 1. The reordering and numerical factorization
            phase = 11;

            std::cout << "[DEBUG] begin decomposition analysis " << std::endl;
            
            pardiso(pt, &maxfct, &mnum, &mtype, &phase, 
                &N, s, is, js, perm,
                &NRHS, iparm, &msglvl, 
                &ddum, &ddum, &error);

            phase = 22;

            std::cout << "[DEBUG] begin decomposition with analysis error: " << error << std::endl;

            pardiso(pt, &maxfct, &mnum, &mtype, &phase, 
                &N, s, is, js, perm,
                &NRHS, iparm, &msglvl, 
                &ddum, &ddum, &error);

            std::cout << "[DEBUG] end for decomposition with error: " << error << std::endl;

        } else { // use the cg solver
            //create hpddm handler
            std::cout << "[DEBUG] construction of hpddm operator " << std::endl;

            S_ = HPDDM::MatrixCSR<cd>(static_cast<int>(n), static_cast<int>(n), SPARSE_INDEX_BASE_ZERO,
                                s, is, js, /*sym=*/false, /*takeOwnership=*/false);
            op = std::make_unique<OverlapOperator>(&S_, S, ovpdescr);
            // op = OverlapOperator(&S_, S, ovpdescr);
        }
    }

    std::cout << "[DEBUG] end of overlap construction " << std::endl;
    
    // create other buffers
    std::vector<cd> B(Bmax+1);
    std::vector<cd> doss(ts.size());
    int nB;
    double tmp;
    
    for(std::size_t it = 0; it < ts.size(); ++it){ // loop over evolved time step
        double dt = (it>0) ? (ts[it] - ts[it-1]) : ts[it];

        for(std::size_t k = 0; k <= Bmax; ++k){
            tmp = boost::math::cyl_bessel_j(k, dt);
            if (fabs(tmp) > btol) {
                B[k] = tmp;
            } else {
                tmp = boost::math::cyl_bessel_j(k+1, dt);
                if (fabs(tmp) > btol) {
                    B[k] = boost::math::cyl_bessel_j(k, dt);
                } else {
                    nB = k;
                    break;
                }
            }
        }

        std::cout << "Iteration " << it << ' ';

        std::vector<cd> Bslice(B.begin(), B.begin() + nB);
        std::cout << "Iteration dt: " << dt << std::endl;

        if (Bslice.size() > 3) {
            if (isoverlap) {
                if (n<1e0) {
                    propagate_direct(H, descr, Bslice, Psi_t, Psi_t, n, nrhs, S, ovpdescr, pt, iparm, forward);
                } else {
                    propagate_cg(H, descr, Bslice, Psi_t, Psi_t, n, nrhs, *op, MPI_COMM_SELF, forward);
                }
            } else {
                propagate(H, descr, Bslice, Psi_t, Psi_t, n, nrhs, forward);
            }
        } // time step is too short, approximate with identity. no else is needed since Psi_t is well initialized.

        // std::cout << "End of output Psi_t" << std::endl;
        // std::cout << "[DEBUG] Psi_in " << Psi_in[0] << std::endl;
        // std::cout << "[DEBUG] Psi_t " << Psi_t[0] << std::endl;
        // compute the trace from Psi_t
        cd caldos;
        cblas_zdotc_sub(nrhs*n, reinterpret_cast<MKL_Complex16*>(Psi_in), 1, reinterpret_cast<MKL_Complex16*>(Psi_t.data()), 1, &caldos);
        
        doss.data()[it] = caldos * static_cast<cd>(n) / static_cast<cd>(nrhs); // here we assume the samples (Psi_in) have already been normalized.
        std::cout << "Cal dos: " << doss.data()[it] << std::endl;
    }

    // end phase
    if (isoverlap) {
        if (n < 1e0) {
            phase = -1;
            pardiso(pt, &maxfct, &mnum, &mtype, &phase, 
                        &N, s, is, js, perm, 
                        &NRHS, iparm, &msglvl, 
                        &ddum, &ddum, &error);
        }
    }
    

    return doss;
}

#endif
#ifndef RSATB_CALC_ISQRTM_VBCSR_HPP
#define RSATB_CALC_ISQRTM_VBCSR_HPP

#include "vbcsr/core/block_csr.hpp"
#include "vbcsr/core/dist_multivector.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <iostream>
#include <algorithm>
#include <set>
#include "graphmf.hpp"

#if defined(RSATB_USE_OPENBLAS) || defined(VBCSR_USE_OPENBLAS)
extern "C" void openblas_set_num_threads(int num_threads);
#endif



using namespace vbcsr;

namespace rsatb {
namespace calc {


template <typename T, typename Kernel = DefaultKernel<T>>
void isqrtm_init(BlockSpMat<T, Kernel>& S_mat, BlockSpMat<T, Kernel>* S_inv_sqrt, bool verbose = false) {
    auto func = [](double x) { return (x > 1e-12) ? std::pow(x, -0.5) : 0.0; };
    vbcsr::graph_matrix_function(S_mat, S_inv_sqrt, std::function<T(double)>(func), "dense", verbose);
}
// Matrix Inverse Square Root using Newton-Schulz Iteration
// Computes S^{-1/2}
template <typename T, typename Kernel = DefaultKernel<T>>
void isqrtm(BlockSpMat<T, Kernel>& S_mat, BlockSpMat<T, Kernel>* S_inv_sqrt, 
            double tol = 1e-7, double sptol = 1e-6, int maxit = 20, std::string init_method = "graph", double factor = 1.0,
            bool verbose = false) {
    
    DistGraph* graph = S_mat.graph;
    int rank = graph->rank;
    
    if (init_method == "graph") {
        isqrtm_init(S_mat, S_inv_sqrt, verbose);
    } else if (init_method == "simple") {
        S_inv_sqrt->shift(T(factor));   
    } else if (init_method == "taylor") {
        S_inv_sqrt->shift(T(1.5));
        S_inv_sqrt->axpy(T(0.5), S_mat);
    } else {
        throw std::runtime_error("Invalid init_method");
    }

    if (maxit == 0) return;

    // ZSZ = (1-0.5D)(1+D)(1-0.5D)=(1+0.5D-0.5D^2)(1-0.5D)=1-0.75D^2+0.25D^3

    // 2. Newton-Schulz Iteration
    // BlockSpMat<T, Kernel> Z = S_inv_sqrt.duplicate(); // Working copy
    BlockSpMat<T, Kernel> tmp1(graph), tmp2(graph);
    
    // Initial Check
    // tmp2 = Z^H * S
    tmp2 = S_inv_sqrt->spmm(S_mat, sptol, true, false); // Z^T * S
    // tmp2 = tmp2 * Z
    tmp2 = tmp2.spmm(*S_inv_sqrt, sptol, false, false);
    // tmp2 = tmp2 - I
    tmp2.shift(T(-1.0));

    // Z, X, XX
    
    // Norm
    double norm = 0.0;
    {
        const auto& norms = tmp2.get_block_norms();
        for(double n : norms) norm = std::max(norm, n);
        double global_norm;
        MPI_Allreduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_MAX, graph->comm);
        norm = global_norm;
    }
    // if (rank == 0) std::cout << "isqrtm: Initial norm=" << norm <<  std::endl;
    
    // Restore tmp2
    tmp2.shift(T(1.0));
    
    double norm_pre = 0.0;
    int count = 0;
    
    while (norm > tol && count < maxit && std::abs(norm_pre - norm) > 0.005 * norm) {
        double density = S_inv_sqrt->get_block_density();
        if (rank == 0 && verbose) {
            std::cout << "Recursive Factorization of S: current error: maxabs norm - " << norm << " with density " << density << std::endl;
        }
        
        if (norm > 100.0) {
            if (rank == 0) std::cout << "isqrtm: DIVERGENCE DETECTED! norm=" << norm << std::endl;
            break;
        }
        
        norm_pre = norm;
        
        // tmp1 = tmp2 * tmp2
        // if (rank == 0 && verbose) std::cout << "isqrtm: Computing tmp1..." << std::endl;
        tmp1 = tmp2.spmm(tmp2, sptol);
        
        // tmp1 = 0.375 * tmp1 - 1.25 * tmp2 + 1.875 * I
        // if (rank == 0 && verbose) std::cout << "isqrtm: Computing tmp1= 0.375 * tmp1 - 1.25 * tmp2." << std::endl;
        tmp1.axpby(T(-1.25), tmp2, T(0.375));
        // if (rank == 0 && verbose) std::cout << "isqrtm: Computing tmp1 = tmp1 + 1.875 * I..." << std::endl;
        tmp1.shift(T(1.875));
        // if (rank == 0 && verbose) std::cout << "isqrtm: Filtering tmp1..." << std::endl;
        tmp1.filter_blocks(sptol);
        
        // Z = Z * tmp1
        // if (rank == 0 && verbose) std::cout << "isqrtm: Computing Z = Z * tmp1..." << std::endl;
        *S_inv_sqrt = S_inv_sqrt->spmm(tmp1, sptol);
        
        // -------------------------
        // *S_inv_sqrt = S_inv_sqrt->spmm(S_inv_sqrt->transpose(), sptol);
        // if (rank == 0 && verbose) std::cout << "isqrtm: Computing S_inv_sqrt = S_inv_sqrt * S_inv_sqrt^H..." << std::endl;
        // S_inv_sqrt->axpby(T(0.5), S_inv_sqrt->transpose(), T(0.5));
        // if (rank == 0 && verbose) std::cout << "isqrtm: Filtering S_inv_sqrt..." << std::endl;
        // S_inv_sqrt->filter_blocks(sptol);

        // Approximate update?
        // tmp3 = tmp1^H * tmp2
        // tmp2 = tmp3 * tmp1
        // if (rank == 0 && verbose) std::cout << "isqrtm: Computing tmp2 = tmp1^H * tmp2..." << std::endl;
        if (norm < 1e-1 && ((count + 1) % 5 != 0)) {
            tmp2 = tmp1.spmm(tmp2, sptol, true, false);
            tmp2 = tmp2.spmm(tmp1, sptol, false, false);

        // //     // if (rank == 0 && verbose) {
        // //     //     std::cout << "Approximate update: " << std::endl;
        // //     // }
        } else {
        //     // Update tmp2 (X_{k+1})
            tmp2 = S_inv_sqrt->spmm(S_mat, sptol, true, false);
            tmp2 = tmp2.spmm(*S_inv_sqrt, sptol, false, false);

            // if (rank == 0 && verbose) {
                // std::cout << "Exact update" << std::endl;
            // }
        }
        
        tmp2.shift(T(-1.0));
        
        // Norm
        norm = 0.0;
        {
            const auto& norms = tmp2.get_block_norms();
            for(double n : norms) norm = std::max(norm, n);
            double global_norm;
            MPI_Allreduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_MAX, graph->comm);
            norm = global_norm;
        }
        
        tmp2.shift(T(1.0));
        
        count++;
    }
    
    double final_density = S_inv_sqrt->get_block_density();
    if (rank == 0 && verbose) {
        std::cout << "Recursive Factorization of S: Converge at maxabs norm - " << norm << " with density " << final_density << std::endl;
    }
}

} // namespace calc
} // namespace rsatb

#endif

#ifndef RSATB_CALC_ISQRTM_HPP
#define RSATB_CALC_ISQRTM_HPP

#include "../backend/block_csr.hpp"
#include "../backend/dbcsr_conversions.hpp"
#include <dbcsr.h>
#include <vector>
#include <cmath>
#include <complex>
#include <iostream>

namespace rsatb {
namespace calc {

using namespace rsatb::backend;

// Matrix Inverse Square Root using Newton-Schulz Iteration
// Computes S^{-1/2}
// Matches functionality of rsatb/core/calc/isqrtm.hpp
template <typename T, typename Kernel = DefaultKernel<T>>
void isqrtm(BlockSpMat<T, Kernel>& S_mat, BlockSpMat<T, Kernel>& S_inv_sqrt, 
            double tol = 1e-7, double sptol = 1e-6, int maxit = 20, double factor = 0.31, 
            bool verbose = false, 
            BlockSpMat<T, Kernel>* H_mat = nullptr, BlockSpMat<T, Kernel>* H_orth = nullptr) {
    
    DistGraph* graph = S_mat.graph;
    MPI_Comm comm = graph->comm;
    int rank = graph->rank;

    if (rank == 0) std::cout << "isqrtm: Entry. factor=" << factor << " tol=" << tol << " maxit=" << maxit << std::endl;

    // 1. Convert S to DBCSR
    dbcsr_matrix S;
    // Determine type
    char mat_type = 'N'; // Treat as general to avoid symmetry storage issues with full input
    // DMat2DBCSR uses 'H' for S.
    
    if (rank == 0) std::cout << "isqrtm: Calling BlockSpMat2DBCSR for S..." << std::endl;
    BlockSpMat2DBCSR(S_mat, &S, mat_type);
    if (rank == 0) std::cout << "isqrtm: S converted." << std::endl;
    
    // 2. Initialize Z (result)
    // Reference initializes Z from mat_isqrt which is shifted by factor.
    // Here we create Z and initialize it.
    dbcsr_matrix Z;
    // We can create Z like S.
    // But we need to initialize it to factor * I.
    
    // Get distribution and sizes from S
    dbcsr_distribution dist;
    c_dbcsr_get_distribution(S, &dist);
    
    int NGlobal = c_dbcsr_nblkrows_total(S);
    if (rank == 0) std::cout << "isqrtm: NGlobal=" << NGlobal << std::endl;
    
    std::vector<int> blksize(NGlobal);
    c_dbcsr_get_col_blk_size(S, blksize.data(), NGlobal);
    
    int data_type;
    if constexpr (std::is_same<T, double>::value) data_type = dbcsr_type_real_8;
    else data_type = dbcsr_type_complex_8;
    
    bool mutable_work = true;
    if (rank == 0) std::cout << "isqrtm: Creating Z..." << std::endl;
    c_dbcsr_create_new(&Z, "MatISQRT", dist, 'N', blksize.data(), NGlobal, blksize.data(), NGlobal, nullptr, &data_type, nullptr, nullptr, &mutable_work, nullptr);
    
    // Initialize Z = factor * I
    // We need to fill diagonal blocks.
    // Iterate blocks owned by this rank in the distribution.
    
    // Helper to fill Identity
    auto fill_identity = [&](dbcsr_matrix& mat, double scale_val) {
        c_dbcsr_reserve_diag_blocks(mat);
        
        int my_rank;
        MPI_Comm group;
        c_dbcsr_get_group(mat, &group);
        MPI_Comm_rank(group, &my_rank);
        
        int my_coords[2];
        MPI_Cart_coords(group, my_rank, 2, my_coords);
        
        int *crows = nullptr, *ccols = nullptr;
        c_dbcsr_distribution_get(dist, &crows, &ccols, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        
        bool sum = false;
        
        for (int bi = 0; bi < NGlobal; ++bi) {
            if (crows[bi] == my_coords[0] && ccols[bi] == my_coords[1]) {
                int bs = blksize[bi];
                if constexpr (std::is_same<T, double>::value) {
                    std::vector<double> blk(bs * bs, 0.0);
                    for (int k = 0; k < bs; ++k) blk[k * bs + k] = scale_val;
                    double scale = 1.0;
                    c_dbcsr_put_block2d_d(mat, bi, bi, blk.data(), bs, bs, &sum, &scale);
                } else {
                    std::vector<std::complex<double>> blk(bs * bs, 0.0);
                    for (int k = 0; k < bs; ++k) blk[k * bs + k] = std::complex<double>(scale_val, 0.0);
                    std::complex<double> scale = 1.0;
                    c_dbcsr_put_block2d_z_api(mat, bi, bi, blk.data(), bs, bs, &sum, &scale);
                }
            }
        }
    };
    
    if (rank == 0) std::cout << "isqrtm: Filling Z identity..." << std::endl;
    fill_identity(Z, factor);
    c_dbcsr_finalize(Z);
    
    // 3. Create Temps
    if (rank == 0) std::cout << "isqrtm: Creating temps..." << std::endl;
    dbcsr_matrix tmp1, tmp2, tmp3, I;
    c_dbcsr_create_new(&tmp1, "tmp1", dist, 'N', blksize.data(), NGlobal, blksize.data(), NGlobal, nullptr, &data_type, nullptr, nullptr, &mutable_work, nullptr);
    c_dbcsr_finalize(tmp1);
    c_dbcsr_create_new(&tmp2, "tmp2", dist, 'N', blksize.data(), NGlobal, blksize.data(), NGlobal, nullptr, &data_type, nullptr, nullptr, &mutable_work, nullptr);
    c_dbcsr_finalize(tmp2);
    c_dbcsr_create_new(&tmp3, "tmp3", dist, 'N', blksize.data(), NGlobal, blksize.data(), NGlobal, nullptr, &data_type, nullptr, nullptr, &mutable_work, nullptr);
    c_dbcsr_finalize(tmp3);
    
    c_dbcsr_create_new(&I, "I", dist, 'N', blksize.data(), NGlobal, blksize.data(), NGlobal, nullptr, &data_type, nullptr, nullptr, &mutable_work, nullptr);
    fill_identity(I, 1.0);
    c_dbcsr_finalize(I);
    
    // 4. Initial Check
    // tmp1 = Z^H * S
    // tmp2 = tmp1 * Z = Z^H * S * Z
    long long int cflop = 0;
    
    // Initialize DBCSR library (idempotent?)
    int io_unit = 0;
    if (rank == 0) std::cout << "isqrtm: Initializing DBCSR lib..." << std::endl;
    c_dbcsr_init_lib(comm, &io_unit);

    if (rank == 0) std::cout << "isqrtm: Initial Check Multiply 1 (Z^H * S)..." << std::endl;
    if (rank == 0) std::cout << "isqrtm: S=" << S << " Z=" << Z << " tmp1=" << tmp1 << std::endl;
    
    if (!S || !Z || !tmp1) {
        if (rank == 0) std::cout << "isqrtm: ERROR: Null matrix pointer!" << std::endl;
        MPI_Abort(comm, 1);
    }

    bool retain_sparsity = false;
    if constexpr (std::is_same<T, double>::value) {
        c_dbcsr_multiply_d('T', 'N', 1.0, Z, S, 0.0, tmp1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
        if (rank == 0) std::cout << "isqrtm: Initial Check Multiply 2 (tmp1 * Z)..." << std::endl;
        c_dbcsr_multiply_d('N', 'N', 1.0, tmp1, Z, 0.0, tmp2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
        
        // tmp2 = tmp2 - I
        if (rank == 0) std::cout << "isqrtm: Subtracting I..." << std::endl;
        c_dbcsr_add_d(tmp2, I, 1.0, -1.0);
    } else {
        c_dbcsr_multiply_z('C', 'N', 1.0, Z, S, 0.0, tmp1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
        if (rank == 0) std::cout << "isqrtm: Initial Check Multiply 2 (tmp1 * Z)..." << std::endl;
        c_dbcsr_multiply_z('N', 'N', 1.0, tmp1, Z, 0.0, tmp2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
        
        if (rank == 0) std::cout << "isqrtm: Subtracting I..." << std::endl;
        c_dbcsr_add_z(tmp2, I, 1.0, -1.0);
    }
    
    double norm = c_dbcsr_maxabs(tmp2);
    if (rank == 0) std::cout << "isqrtm: Initial norm=" << norm << std::endl;
    
    // Restore tmp2 for iteration (tmp2 = Z^H S Z - I + I = Z^H S Z)
    if constexpr (std::is_same<T, double>::value) c_dbcsr_add_d(tmp2, I, 1.0, 1.0);
    else c_dbcsr_add_z(tmp2, I, 1.0, 1.0);
    
    double norm_pre = 0.0;
    int count = 0;
    double st_sptol = 0.1 * sptol;
    
    // 5. Newton-Schulz Loop
    if (rank == 0) std::cout << "isqrtm: Starting Loop..." << std::endl;
    while (norm > tol && count < maxit && std::abs(norm_pre - norm) > 0.005 * norm) {
        if (rank == 0 && verbose) {
            std::cout << "Recursive Factorization of S: current error: maxabs norm - " << norm << std::endl;
        }
        
        if (norm > 100.0) {
            if (rank == 0) std::cout << "isqrtm: DIVERGENCE DETECTED! norm=" << norm << std::endl;
            break;
        }
        
        if (norm > 100.0) {
            if (rank == 0) std::cout << "isqrtm: DIVERGENCE DETECTED! norm=" << norm << std::endl;
            break;
        }
        
        norm_pre = norm;
        
        // tmp1 = tmp2 * tmp2
        if constexpr (std::is_same<T, double>::value) {
            c_dbcsr_multiply_d('N', 'N', 1.0, tmp2, tmp2, 0.0, tmp1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
            
            // update Z
            // tmp1 = 0.375 * tmp2 - 1.25 * tmp1
            c_dbcsr_add_d(tmp1, tmp2, 0.375, -1.25);
            // tmp1 = tmp1 + 1.875 * I
            c_dbcsr_add_d(tmp1, I, 1.0, 1.875);
            
            c_dbcsr_filter(tmp1, &st_sptol, nullptr, nullptr, nullptr);
            
            // tmp3 = Z * tmp1
            c_dbcsr_multiply_d('N', 'N', 1.0, Z, tmp1, 0.0, tmp3, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
            
            // Z = Z + tmp3 (Wait, reference says: c_dbcsr_add_z(Z, tmp3, 0.0, 1.0); which means Z = tmp3)
            // Yes, Z = Z * P_k
            c_dbcsr_add_d(Z, tmp3, 0.0, 1.0);
            c_dbcsr_filter(Z, &st_sptol, nullptr, nullptr, nullptr);
            
            // Update tmp2 (next X)
            if (norm < 1e-1 && ((count + 1) % 5 != 0)) {
                // Approximate update?
                // tmp3 = tmp1^H * tmp2
                // tmp2 = tmp3 * tmp1
                // This looks like X_{k+1} = P_k^H X_k P_k
                c_dbcsr_multiply_d('T', 'N', 1.0, tmp1, tmp2, 0.0, tmp3, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
                c_dbcsr_multiply_d('N', 'N', 1.0, tmp3, tmp1, 0.0, tmp2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
            } else {
                // Full recompute: tmp2 = Z^H * S * Z
                c_dbcsr_multiply_d('T', 'N', 1.0, Z, S, 0.0, tmp3, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
                c_dbcsr_multiply_d('N', 'N', 1.0, tmp3, Z, 0.0, tmp2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
            }
            
            c_dbcsr_add_d(tmp2, I, 1.0, -1.0);
            norm = c_dbcsr_maxabs(tmp2);
            c_dbcsr_add_d(tmp2, I, 1.0, 1.0);
            
        } else {
            c_dbcsr_multiply_z('N', 'N', 1.0, tmp2, tmp2, 0.0, tmp1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
            
            c_dbcsr_add_z(tmp1, tmp2, 0.375, -1.25);
            c_dbcsr_add_z(tmp1, I, 1.0, 1.875);
            
            c_dbcsr_filter(tmp1, &st_sptol, nullptr, nullptr, nullptr);
            
            c_dbcsr_multiply_z('N', 'N', 1.0, Z, tmp1, 0.0, tmp3, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
            
            c_dbcsr_add_z(Z, tmp3, 0.0, 1.0);
            c_dbcsr_filter(Z, &st_sptol, nullptr, nullptr, nullptr);
            
            if (norm < 1e-1 && ((count + 1) % 5 != 0)) {
                c_dbcsr_multiply_z('C', 'N', 1.0, tmp1, tmp2, 0.0, tmp3, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
                c_dbcsr_multiply_z('N', 'N', 1.0, tmp3, tmp1, 0.0, tmp2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
            } else {
                c_dbcsr_multiply_z('C', 'N', 1.0, Z, S, 0.0, tmp3, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
                c_dbcsr_multiply_z('N', 'N', 1.0, tmp3, Z, 0.0, tmp2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
            }
            
            c_dbcsr_add_z(tmp2, I, 1.0, -1.0);
            norm = c_dbcsr_maxabs(tmp2);
            c_dbcsr_add_z(tmp2, I, 1.0, 1.0);
        }
        
        count++;
    }
    
    if (rank == 0 && verbose) {
        std::cout << "Recursive Factorization of S: Converge at maxabs norm - " << norm << std::endl;
    }
    
    // 6. Orthogonalization (Optional)
    if (H_mat != nullptr && H_orth != nullptr) {
        if (rank == 0 && verbose) std::cout << "Performing orthogonalization in isqrtm..." << std::endl;
        
        dbcsr_matrix H, H_new, tmp_H;
        BlockSpMat2DBCSR(*H_mat, &H, 'H'); // Assume H is Hermitian
        
        c_dbcsr_create_new(&tmp_H, "tmp_H", dist, 'N', blksize.data(), NGlobal, blksize.data(), NGlobal, nullptr, &data_type, nullptr, nullptr, &mutable_work, nullptr);
        c_dbcsr_finalize(tmp_H);
        
        c_dbcsr_create_new(&H_new, "H_new", dist, 'N', blksize.data(), NGlobal, blksize.data(), NGlobal, nullptr, &data_type, nullptr, nullptr, &mutable_work, nullptr);
        c_dbcsr_finalize(H_new);
        
        if constexpr (std::is_same<T, double>::value) {
            // tmp_H = H * Z
            c_dbcsr_multiply_d('N', 'N', 1.0, H, Z, 0.0, tmp_H, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
            // H_new = Z^H * tmp_H
            c_dbcsr_multiply_d('T', 'N', 1.0, Z, tmp_H, 0.0, H_new, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
        } else {
            c_dbcsr_multiply_z('N', 'N', 1.0, H, Z, 0.0, tmp_H, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
            c_dbcsr_multiply_z('C', 'N', 1.0, Z, tmp_H, 0.0, H_new, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &retain_sparsity, &sptol, &cflop);
        }
        
        DBCSR2BlockSpMat(H_new, *H_orth);
        
        c_dbcsr_release(&H);
        c_dbcsr_release(&H_new);
        c_dbcsr_release(&tmp_H);
    }
    
    if (rank == 0) std::cout << "isqrtm: Output Result..." << std::endl;
    // 7. Output Result
    DBCSR2BlockSpMat(Z, S_inv_sqrt);
    
    if (rank == 0) std::cout << "isqrtm: Cleanup Z..." << std::endl;
    c_dbcsr_release(&Z);
    if (rank == 0) std::cout << "isqrtm: Cleanup tmp1..." << std::endl;
    c_dbcsr_release(&tmp1);
    if (rank == 0) std::cout << "isqrtm: Cleanup tmp2..." << std::endl;
    c_dbcsr_release(&tmp2);
    if (rank == 0) std::cout << "isqrtm: Cleanup tmp3..." << std::endl;
    c_dbcsr_release(&tmp3);
    if (rank == 0) std::cout << "isqrtm: Cleanup I..." << std::endl;
    c_dbcsr_release(&I);
    
    if (rank == 0) std::cout << "isqrtm: Cleanup S..." << std::endl;
    c_dbcsr_release(&S);
    
    if (rank == 0) std::cout << "isqrtm: Cleanup Done." << std::endl;
    // dist is managed by S? No, get_distribution returns pointer to internal?
    // Reference calls c_dbcsr_distribution_release(&dist) only if it created it?
    // Reference: c_dbcsr_get_distribution(S, &dist); ... c_dbcsr_distribution_release(&dist);
    // Wait, get_distribution usually returns a handle. Does it increment refcount?
    // The reference code releases it. So I should too.
    // But `dist` was obtained from `S`.
    // Let's check `dbcsr_capi.h` or usage.
    // If I release it, does S become invalid?
    // S is released before dist.
    // I'll follow reference.
    // c_dbcsr_distribution_release(&dist); // Reference does this.
}

} // namespace calc
} // namespace rsatb

#endif

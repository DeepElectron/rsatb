#include "../image_container.hpp"
#include <mpi.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

using namespace rsatb::backend;

void test_image_container() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) std::cout << "Starting ImageContainer Test..." << std::endl;

    // 1. Setup AtomicData
    // 2 atoms: 0 at (0,0,0), 1 at (0.5, 0.5, 0.5)
    // Cell: 10, 10, 10
    
    // Partition atoms:
    // Rank 0: Atom 0
    // Rank 1: Atom 1
    // If size=1, Rank 0 has both.
    
    int my_n_atom = (rank < 2) ? 1 : 0;
    if (size == 1) my_n_atom = 2;
    
    int atom_offset = (rank == 1) ? 1 : 0;
    
    std::vector<int> atom_index;
    std::vector<int> atom_type;
    std::vector<double> pos;
    
    if (size == 1) {
        atom_index = {0, 1};
        atom_type = {0, 0};
        pos = {0.0, 0.0, 0.0, 5.0, 5.0, 5.0};
    } else {
        if (rank == 0) {
            atom_index = {0};
            atom_type = {0};
            pos = {0.0, 0.0, 0.0};
        } else if (rank == 1) {
            atom_index = {1};
            atom_type = {0};
            pos = {5.0, 5.0, 5.0};
        }
    }

    std::vector<int> type_norb = {1}; // 1 orbital per atom
    std::vector<double> cell = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
    
    // Edges:
    // 0->1 R=(0,0,0)
    // 0->1 R=(1,0,0)
    
    std::vector<int> edge_index;
    std::vector<int> edge_shift;
    
    // Edges must be provided by owner of source.
    // Src 0 is on Rank 0.
    if (rank == 0) {
        edge_index = {0, 1, 0, 1};
        edge_shift = {0, 0, 0, 1, 0, 0};
    }
    
    int n_edge = edge_index.size() / 2;
    
    AtomicData* data = new AtomicData(
        my_n_atom, 2, atom_offset, n_edge, 2,
        atom_index.data(), atom_type.data(), edge_index.data(), type_norb.data(), edge_shift.data(),
        cell.data(), pos.data(), MPI_COMM_WORLD
    );
    
    if (rank == 0) std::cout << "AtomicData created." << std::endl;

    // 2. Create ImageContainer
    ImageContainer<double>* img = new ImageContainer<double>(data);
    
    if (rank == 0) std::cout << "ImageContainer created." << std::endl;

    // Verify graphs
    // Should have R=(0,0,0) and R=(1,0,0)
    std::vector<int> R0 = {0, 0, 0};
    std::vector<int> R1 = {1, 0, 0};
    
    assert(img->image_graphs.count(R0));
    assert(img->image_graphs.count(R1));
    assert(img->image_graphs.size() == 2);
    
    if (rank == 0) std::cout << "Graphs verified." << std::endl;

    // 3. Add blocks
    // Block size 1x1
    double val0 = 1.0;
    double val1 = 2.0;
    
    // Add blocks. Only owner of the row (src) should add?
    // Or anyone can add remote blocks?
    // ImageContainer::add_block handles remote blocks.
    // But we should avoid double adding.
    // In this test, we add for 0->1. Src is 0.
    // Rank 0 owns 0.
    // So Rank 0 adds.
    
    if (rank == 0) {
        img->add_block(R0, 0, 1, &val0, 1, 1);
        img->add_block(R1, 0, 1, &val1, 1, 1);
    }
    
    img->assemble();
    
    if (rank == 0) std::cout << "Blocks added." << std::endl;

    // 4. Sample K
    // K = (0, 0, 0) -> Phase = 1
    // Result(0, 1) = 1.0 + 2.0 = 3.0
    std::vector<double> K0 = {0.0, 0.0, 0.0};
    auto* res0 = img->sample_k(K0, PhaseConvention::R_ONLY);
    
    // Check result
    // 0->1 is owned by Rank 0.
    if (rank == 0) {
        int l_row = data->graph->global_to_local.at(0);
        int l_col = data->graph->global_to_local.at(1);
        
        bool found = false;
        int start = res0->row_ptr[l_row];
        int end = res0->row_ptr[l_row+1];
        for (int k = start; k < end; ++k) {
            if (res0->col_ind[k] == l_col) {
                std::complex<double> val = res0->val[res0->blk_ptr[k]];
                if (rank == 0) std::cout << "K=0 Val: " << val << std::endl;
                assert(std::abs(val.real() - 3.0) < 1e-9);
                found = true;
            }
        }
        assert(found);
    }
    delete res0;

    // K = (0.5, 0, 0) -> Phase R=1 is exp(i * 2pi * 0.5 * 1) = exp(i*pi) = -1
    // Result(0, 1) = 1.0 + 2.0 * (-1) = -1.0
    std::vector<double> K1 = {0.5, 0.0, 0.0};
    auto* res1 = img->sample_k(K1, PhaseConvention::R_ONLY);
    
    if (rank == 0) {
        int l_row = data->graph->global_to_local.at(0);
        int l_col = data->graph->global_to_local.at(1);
        
        bool found = false;
        int start = res1->row_ptr[l_row];
        int end = res1->row_ptr[l_row+1];
        for (int k = start; k < end; ++k) {
            if (res1->col_ind[k] == l_col) {
                std::complex<double> val = res1->val[res1->blk_ptr[k]];
                if (rank == 0) std::cout << "K=0.5 Val: " << val << std::endl;
                assert(std::abs(val.real() - (-1.0)) < 1e-9);
                found = true;
            }
        }
        assert(found);
    }
    delete res1;
    
    // Test R_AND_POSITION
    // r0 = (0,0,0), r1 = (5,5,5) = (0.5, 0.5, 0.5) fractional
    // dr = (0.5, 0.5, 0.5)
    // K = (0.5, 0, 0)
    // Phase pos = exp(i * 2pi * 0.5 * 0.5) = exp(i * pi/2) = i
    // R=0: phase = 1 * i = i. Val = 1.0 * i = i
    // R=1: phase = -1 * i = -i. Val = 2.0 * -i = -2i
    // Total = i - 2i = -i
    
    auto* res2 = img->sample_k(K1, PhaseConvention::R_AND_POSITION);
    
    if (rank == 0) {
        int l_row = data->graph->global_to_local.at(0);
        int l_col = data->graph->global_to_local.at(1);
        
        bool found = false;
        int start = res2->row_ptr[l_row];
        int end = res2->row_ptr[l_row+1];
        for (int k = start; k < end; ++k) {
            if (res2->col_ind[k] == l_col) {
                std::complex<double> val = res2->val[res2->blk_ptr[k]];
                if (rank == 0) std::cout << "K=0.5 Pos Val: " << val << std::endl;
                assert(std::abs(val.real()) < 1e-9);
                assert(std::abs(val.imag() - (-1.0)) < 1e-9);
                found = true;
            }
        }
        assert(found);
    }
    delete res2;

    delete img;
    delete data;
    
    if (rank == 0) std::cout << "Single Element Test passed!" << std::endl;
}

void test_multielement_multiorbital() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) std::cout << "\nStarting Multi-Element/Multi-Orbital Test..." << std::endl;

    // Setup AtomicData
    // 2 atoms:
    // Atom 0: Type 0, Pos (0,0,0), Norb 2
    // Atom 1: Type 1, Pos (2,2,2), Norb 3
    // Cell: 4, 4, 4
    
    int n_atom = 2;
    int atom_offset = 0; 
    
    int my_n_atom = (rank < 2) ? 1 : 0; // Support up to 2 ranks. If >2, others empty.
    if (size == 1) my_n_atom = 2;
    
    std::vector<int> atom_index;
    std::vector<int> atom_type;
    std::vector<double> pos;
    
    if (size == 1) {
        atom_index = {0, 1};
        atom_type = {0, 1};
        pos = {0.0, 0.0, 0.0, 2.0, 2.0, 2.0};
    } else {
        if (rank == 0) {
            atom_index = {0};
            atom_type = {0};
            pos = {0.0, 0.0, 0.0};
        } else if (rank == 1) {
            atom_index = {1};
            atom_type = {1};
            pos = {2.0, 2.0, 2.0};
        }
    }
    
    std::vector<int> type_norb = {2, 3}; // Type 0: 2 orbitals, Type 1: 3 orbitals
    std::vector<double> cell = {4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0};
    
    // Edges:
    // 0 -> 1 with R=(0,0,0)
    // 0 -> 1 with R=(1,0,0)
    // 1 -> 0 with R=(0,0,0) (Reverse)
    
    std::vector<int> edge_index;
    std::vector<int> edge_shift;
    
    if (rank == 0) {
        // 0->1 R=(0,0,0)
        edge_index.push_back(0); edge_index.push_back(1);
        edge_shift.push_back(0); edge_shift.push_back(0); edge_shift.push_back(0);
        
        // 0->1 R=(1,0,0)
        edge_index.push_back(0); edge_index.push_back(1);
        edge_shift.push_back(1); edge_shift.push_back(0); edge_shift.push_back(0);
    }
    
    if ((size == 1) || (rank == 1)) {
        // 1->0 R=(0,0,0)
        edge_index.push_back(1); edge_index.push_back(0);
        edge_shift.push_back(0); edge_shift.push_back(0); edge_shift.push_back(0);
    }
    
    int n_edge = edge_index.size() / 2;
    
    AtomicData* data = new AtomicData(
        my_n_atom, 2, (rank == 1 ? 1 : 0), n_edge, 3,
        atom_index.data(), atom_type.data(), edge_index.data(), type_norb.data(), edge_shift.data(),
        cell.data(), pos.data(), MPI_COMM_WORLD
    );
    
    if (rank == 0) std::cout << "AtomicData created." << std::endl;
    
    ImageContainer<double>* img = new ImageContainer<double>(data);
    
    // Add blocks
    // 0->1 (Size 2x3) for R=(0,0,0)
    // 0->1 (Size 2x3) for R=(1,0,0)
    // 1->0 (Size 3x2) for R=(0,0,0)
    
    // Data values:
    // 0->1 R0: All 1.0
    // 0->1 R1: All 2.0
    // 1->0 R0: All 3.0
    
    std::vector<double> block_01_r0(2*3, 1.0);
    std::vector<double> block_01_r1(2*3, 2.0);
    std::vector<double> block_10_r0(3*2, 3.0);
    
    std::vector<int> R0 = {0, 0, 0};
    std::vector<int> R1 = {1, 0, 0};
    
    if (rank == 0) {
        img->add_block(R0, 0, 1, block_01_r0.data(), 2, 3);
        img->add_block(R1, 0, 1, block_01_r1.data(), 2, 3);
    }
    
    if ((size == 1) || (rank == 1)) {
        img->add_block(R0, 1, 0, block_10_r0.data(), 3, 2);
    }
    
    img->assemble();
    
    if (rank == 0) std::cout << "Blocks added and assembled." << std::endl;
    
    // Test Sample K
    // K = (0.25, 0, 0)
    std::vector<double> K = {0.25, 0.0, 0.0};
    
    // Convention R_ONLY
    // Phase R0 = 1
    // Phase R1 = exp(i * 2pi * 0.25 * 1) = exp(i * pi/2) = i
    
    // Expected 0->1:
    // Block = Block_R0 * 1 + Block_R1 * i
    //       = 1.0 + 2.0i (for all elements)
    
    // Expected 1->0:
    // Block = Block_R0 * 1 = 3.0 (for all elements)
    
    auto* res = img->sample_k(K, PhaseConvention::R_ONLY);
    
    // Verify 0->1 (Owned by Rank 0)
    if (rank == 0) {
        int l_row = data->graph->global_to_local.at(0);
        int l_col = data->graph->global_to_local.at(1); // 1 is ghost on rank 0 (if np=2)
        
        bool found = false;
        int start = res->row_ptr[l_row];
        int end = res->row_ptr[l_row+1];
        for (int k = start; k < end; ++k) {
            if (res->col_ind[k] == l_col) {
                size_t offset = res->blk_ptr[k];
                for (int i=0; i<2*3; ++i) {
                    std::complex<double> val = res->val[offset + i];
                    if (i == 0) std::cout << "0->1 Val: " << val << std::endl;
                    assert(std::abs(val.real() - 1.0) < 1e-9);
                    assert(std::abs(val.imag() - 2.0) < 1e-9);
                }
                found = true;
            }
        }
        assert(found);
    }
    
    // Verify 1->0 (Owned by Rank 1, or Rank 0 if serial)
    if ((size == 1) || (rank == 1)) {
        int l_row = data->graph->global_to_local.at(1);
        int l_col = data->graph->global_to_local.at(0);
        
        bool found = false;
        int start = res->row_ptr[l_row];
        int end = res->row_ptr[l_row+1];
        for (int k = start; k < end; ++k) {
            if (res->col_ind[k] == l_col) {
                size_t offset = res->blk_ptr[k];
                for (int i=0; i<3*2; ++i) {
                    std::complex<double> val = res->val[offset + i];
                    if (i == 0) std::cout << "1->0 Val: " << val << std::endl;
                    assert(std::abs(val.real() - 3.0) < 1e-9);
                    assert(std::abs(val.imag() - 0.0) < 1e-9);
                }
                found = true;
            }
        }
        assert(found);
    }
    
    delete res;
    
    // Convention R_AND_POSITION
    // r0 = (0,0,0), r1 = (2,2,2)
    // Cell = 4,4,4 -> r1_frac = (0.5, 0.5, 0.5)
    
    // 0->1: dr = r1 - r0 = (0.5, 0.5, 0.5)
    // Phase pos = exp(i * 2pi * K * dr) = exp(i * 2pi * 0.25 * 0.5) = exp(i * pi/4)
    // exp(i * pi/4) = 1/sqrt(2) + i/sqrt(2)
    
    // R0 (0->1): Phase = 1 * exp(i pi/4)
    // R1 (0->1): Phase = i * exp(i pi/4) = exp(i * 3pi/4)
    
    // Val = 1.0 * exp(i pi/4) + 2.0 * exp(i 3pi/4)
    //     = (1/s2 + i/s2) + 2 * (-1/s2 + i/s2)
    //     = 1/s2 - 2/s2 + i(1/s2 + 2/s2)
    //     = -1/s2 + 3i/s2
    //     = -0.70710678 + 2.1213203i
    
    // 1->0: dr = r0 - r1 = (-0.5, -0.5, -0.5)
    // Phase pos = exp(i * 2pi * 0.25 * -0.5) = exp(-i pi/4)
    // R0 (1->0): Phase = 1 * exp(-i pi/4)
    // Val = 3.0 * exp(-i pi/4) = 3/s2 - 3i/s2
    
    auto* res2 = img->sample_k(K, PhaseConvention::R_AND_POSITION);
    
    double s2 = std::sqrt(2.0);
    
    if (rank == 0) {
        int l_row = data->graph->global_to_local.at(0);
        int l_col = data->graph->global_to_local.at(1);
        
        bool found = false;
        int start = res2->row_ptr[l_row];
        int end = res2->row_ptr[l_row+1];
        for (int k = start; k < end; ++k) {
            if (res2->col_ind[k] == l_col) {
                size_t offset = res2->blk_ptr[k];
                std::complex<double> val = res2->val[offset]; // Check first element
                std::cout << "0->1 Pos Val: " << val << " Expected: " << -1.0/s2 << " + " << 3.0/s2 << "i" << std::endl;
                
                assert(std::abs(val.real() - (-1.0/s2)) < 1e-9);
                assert(std::abs(val.imag() - (3.0/s2)) < 1e-9);
                found = true;
            }
        }
        assert(found);
    }
    
    if ((size == 1) || (rank == 1)) {
        int l_row = data->graph->global_to_local.at(1);
        int l_col = data->graph->global_to_local.at(0);
        
        bool found = false;
        int start = res2->row_ptr[l_row];
        int end = res2->row_ptr[l_row+1];
        for (int k = start; k < end; ++k) {
            if (res2->col_ind[k] == l_col) {
                size_t offset = res2->blk_ptr[k];
                std::complex<double> val = res2->val[offset];
                std::cout << "1->0 Pos Val: " << val << " Expected: " << 3.0/s2 << " - " << 3.0/s2 << "i" << std::endl;
                
                assert(std::abs(val.real() - (3.0/s2)) < 1e-9);
                assert(std::abs(val.imag() - (-3.0/s2)) < 1e-9);
                found = true;
            }
        }
        assert(found);
    }
    
    delete res2;
    delete img;
    delete data;
    
    if (rank == 0) std::cout << "Multi-Element/Multi-Orbital Test passed!" << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    test_image_container();
    test_multielement_multiorbital();
    MPI_Finalize();
    return 0;
}

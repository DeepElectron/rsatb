#include "../../model/lcao/op2c_engine.hpp"
#include "../../../test_data_path.hpp"

#include <fstream>
#include "vbcsr/core/atomic/atomic_data.hpp"
#include <gtest/gtest.h>
#include <complex>
#include <vector>
#include <string>
#include <mpi.h>

using namespace rsatb;
using namespace rsatb::model::lcao;

// Reusing setup from test_hr2hk.cpp
class GlobalEnv : public ::testing::Environment {
public:
    int rank, size;
    // Data from test_hr2hk.cpp
    int n_atom=1, n_edge=292;
    int atom_type1[1] = {0};
    int atom_type2[1] = {1};
    int type_norb[2] = {13, 13};
    double cell[9] = {2.52685 , 0., 0., 1.263425, 2.188316, 0., 1.263425, 0.729439, 2.063165};
    double pos1[3] = {2.52685   , 1.4588775 , 1.0315825};
    double pos2[3] = {3.790275  , 2.18831625, 1.54737375};
    int atom_index1 = 0;
    int atom_index2 = 1;
    int atom_offset1=0;
    int atom_offset2=1;
    
    // Edges (truncated for brevity in this snippet, but should be full for real test)
    // We will use the full arrays from test_hr2hk.cpp if possible, or a subset if it works.
    // For correctness, we should use the full arrays.
    // I will copy the full arrays from test_hr2hk.cpp.
    
    std::vector<int> edge_index1_vec;
    std::vector<int> edge_shift_vec1_vec;
    std::vector<int> edge_index2_vec;
    std::vector<int> edge_shift_vec2_vec;

    GlobalEnv() {
        // Initialize vectors with data from test_hr2hk.cpp
        // (I will populate these in the constructor or use the raw arrays if I copy them)
    }

    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
};

// We need the full data arrays. 
// I will include them directly in the test file for simplicity, copying from test_hr2hk.cpp.

int edge_index1[584] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0,
      0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1,
      0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1,
      0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1,
      0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1,
      0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1,
      0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
      0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1,
      0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1,
      0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
      0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0,
      0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
      0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1,
      0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
      0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
      0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
      0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1,
      0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
      0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
      0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
      
int edge_shift_vec1[876] = {
      1,  2, -3,  1,  2, -3,  0, -2,  3,  0, -2,  3, -1,  1, -3,  0,  1,
      -3,  1,  1, -3,  2,  1, -3,  2,  1, -3, -1, -1,  3, -1, -1,  3,  0,
      -1,  3, -1, -2,  3,  0,  2, -3, -2, -1,  3,  1,  1, -3,  0,  1,  2,
      -1, -1, -2,  1, -2, -2,  2, -2, -2,  3, -2, -2, -2,  2,  2,  0, -2,
      -2, -1, -2,  3,  0,  0, -3,  1,  0, -3,  2,  0, -3, -2,  0,  3, -2,
      0,  3, -1,  0,  3,  1, -1, -3,  0,  2, -3, -1,  1, -2,  0,  1, -2,
      1,  1, -2,  1,  1, -2,  0, -1,  2,  0, -1,  2,  1, -1,  2,  1, -1,
      2,  3,  0, -2, -3,  0,  2, -3,  0,  2, -3,  1,  2, -1,  0, -2,  0,
      0, -2,  1,  0, -2,  2,  0, -2, -1, -2,  2, -1, -2,  2, -1, -1,  2,
      -2,  2, -2, -1,  2, -2,  0,  2, -2,  0,  2, -2,  1, -2,  2,  1, -2,
      2,  2, -2,  2,  2,  1, -2, -3, -1,  2, -2, -1,  2, -2, -1,  2, -2,
      0,  2, -2,  1, -2,  0,  2,  1, -1, -2, -1,  1, -3, -1,  2, -3, -1,
      1, -3,  2,  0,  3, -2, -1, -3,  2,  0, -3,  2,  0, -3,  2,  0, -2,
      2, -2,  3, -2, -1,  3, -2, -1,  3, -2,  1,  2, -2, -2, -2,  2, -2,
      -2,  2,  2, -1, -1, -1,  1,  1, -1,  1,  1,  0,  1,  1,  0,  1,  1,
      1,  1,  1, -2, -1, -1,  0, -2, -1,  1, -2, -1,  2, -2, -1,  3, -2,
      -1,  3, -2, -1, -2,  2,  1, -2,  2,  1, -1,  2,  1, -1,  2,  1,  0,
      0, -1,  1,  0, -1,  1,  0, -1,  0,  0,  1,  0,  0,  1,  1,  0,  1,
      1,  0,  1,  2,  0,  1,  3, -1, -1, -3,  1,  1, -3,  1,  1, -3,  2,
      1, -1, -1, -1,  0, -1, -1,  1, -1, -1,  2, -1, -1, -2,  1, -1, -1,
      1, -1,  0,  1, -1,  0,  1, -1,  1, -1,  1,  1, -1,  1,  2, -1,  1,
      2, -1,  1,  2,  0, -1, -3,  0,  1, -3,  0,  1, -2,  0,  1, -2,  0,
      1, -2,  1,  1, -2,  0, -1, -1,  0, -1,  0, -1,  1, -3,  2, -1, -2,
      2, -1, -1,  2, -1, -1,  2, -1,  2, -2,  1,  2, -2,  1,  1,  1, -1,
      2,  1, -1, -3, -1,  1, -2, -1,  1, -2, -1,  1, -1, -1,  1, -1, -1,
      1, -1,  0,  1, -3,  1, -1, -1,  3, -1, -1, -3,  1,  0, -3,  1,  0,
      -3,  1,  1, -3,  1,  1, -3,  1,  1, -2,  1, -2,  3, -1, -2,  3, -1,
      0,  2, -1,  1,  2, -1, -2, -2,  1, -1, -2,  1, -1, -2,  1,  0, -2,
      1,  0, -2,  1,  1, -2,  0,  2, -2,  0,  2, -2,  0, -1,  2,  0, -1,
      2,  0,  0,  2,  0,  0,  2,  0,  1,  2,  0, -2, -2,  0,  0, -3,  0,
      1, -3,  0,  2, -3,  0, -2,  3,  0, -2,  3,  0, -1,  3,  0,  2, -3,
      1, -2,  2,  0, -2, -1,  0, -1, -1,  0,  0, -1,  0,  1, -1,  0,  1,
      -1,  0,  0,  1,  0,  0,  1,  0,  1,  1,  0,  1,  1,  0,  2,  1,  0,
      3, -2,  0, -3,  2,  0, -3,  2,  0, -1, -2,  0,  0, -2,  0, -1,  0,
      0, -2,  0,  0, -1,  1,  0, -3,  0,  0, -2,  0,  0, -1,  0,  0,  0,
      0,  0,  1,  0,  0,  1,  0,  0,  2,  0,  0,  2,  0,  0,  2, -1,  0,
      -3,  1,  0, -3,  1,  0, -2,  1,  0, -2,  1,  0,  2, -3,  0,  1, -3,
      0,  0,  3, -1, -1,  3, -1,  2, -3,  1,  3, -2,  0, -2,  2,  0,  1,
      -2,  0,  0, -2,  0, -1, -2,  0,  3, -1,  0,  2, -1,  0, -1,  1,  0,
      0, -1,  0, -1, -1,  0, -2, -1,  0, -1,  0,  1,  0,  0, -1, -1,  0,
      -1, -2,  0, -1,  3, -1, -1,  1,  1, -1,  0, -1,  1, -1,  1, -1, -2,
      1, -1,  3,  0, -1,  2,  0, -1,  1,  2, -1,  0,  2, -1,  1, -2,  1,
      -2,  2, -1,  2,  1, -1,  3,  0, -2,  2,  2, -2,  1,  2, -2,  0, -2,
      2, -1,  2, -2, -2,  2, -2,  2, -2, -1,  1, -2, -1,  0, -2, -1,  0,
      3, -2,  1, -3,  2, -2,  1,  1,  1, -1, -1,  0, -1, -1, -1, -1, -1,
      -3,  2,  1,  0,  1, -3,  2,  0, -3,  2, -1, -2,  1, -1, -2,  0, -1,
      -2,  2, -2, -2,  1,  0, -3, -2,  0,  2,  1,  0, -2,  0,  0, -2, -1,
      0, -2, -3,  1,  2,  2,  1, -2, -1, -1,  2,  0,  1, -2, -1,  1, -2,
      2,  0, -2, -1,  0,  2, -1,  0,  2,  0,  0,  2,  0,  0,  2,  1,  0,
      2, -2,  0, -2,  0, -1, -2,  1, -1, -2,  2, -1, -2,  3, -1, -2,  3,
      -1, -2, -2,  1,  2, -2,  1,  2, -1,  1,  2, -1,  1,  2, -1,  2, -3,
      -2, -1,  3, -2, -2,  3,  2, -1, -3};
      
int edge_index2[584] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
      1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };
    
int edge_shift_vec2[876] = {
      2,  0, -3,  1,  0, -3,  1,  1, -3,  0,  1, -3,  2,  1, -3,  0,  2,
      -3,  1,  2, -3,  2, -2, -2,  2, -1, -2,  1, -1, -2,  0, -1, -2,  3,
      -1, -2,  1,  0, -2,  0,  0, -2, -1,  0, -2,  3,  0, -2, -1, -2,  3,
      0,  2, -3,  1, -1,  3,  0, -1,  3, -1, -1,  3, -2, -1,  3,  1,  1,
      -3,  0,  0,  3, -1,  0,  3, -2,  0,  3,  2, -1, -2,  1, -1, -2,  1,
      1,  2, -1,  2,  2, -2,  2,  2, -3,  2,  2,  0,  2,  2,  1,  2, -3,
      2,  2, -3,  2,  1, -3,  1, -2,  3,  0, -2,  3,  2,  0, -3, -1,  1,
      3, -2,  1,  3, -2,  0,  3, -1,  0,  3, -1, -1,  3,  0, -1,  3, -2,
      -1,  3,  0, -2,  3, -1, -2,  3, -2,  2,  2, -2,  1,  2, -1,  1,  2,
      0,  1,  2, -3,  1,  2,  1,  0,  2,  0,  0,  2, -1,  0,  2, -2,  0,
      2,  1,  0, -2,  0,  0, -2,  2,  0,  2,  0,  1,  2, -1,  1,  2, -2,
      1,  2, -3,  1,  2,  2,  1, -2,  2,  0, -2,  2, -1,  2,  1, -1,  2,
      0, -1,  2, -1, -1,  2,  0,  1, -2, -1,  1, -2, -3,  0,  2,  3,  0,
      -2,  3, -1, -2,  1, -3,  2, -1, -2,  2,  2,  2, -2,  1,  2, -2,  1,
      1, -2,  2, -2,  2,  1, -2,  2,  0, -2,  2, -1,  2, -2, -2, -1,  2,
      3,  1, -2, -1,  0,  2,  0,  0,  2,  1,  0,  2, -3,  0,  2, -2,  0,
      2,  0, -1,  2,  1, -1,  2, -2, -1,  2, -1, -1,  2,  1, -2,  2,  2,
      -2,  2, -2, -2,  2, -1, -2,  2,  0, -2,  2,  0, -3,  2,  1, -3,  2,
      2, -2, -1,  1, -2, -1,  1,  2,  1, -1,  3,  1, -2,  3,  1, -1,  3,
      -2,  0, -3,  2,  1,  3, -2,  0,  3, -2,  0,  2, -2,  2, -3,  2,  1,
      1,  1,  0,  1,  1, -1,  1,  1, -2,  1,  1,  1, -1, -1,  0, -1, -1,
      2,  1,  1,  0,  2,  1, -1,  2,  1, -2,  2,  1, -3,  2,  1,  2,  0,
      -1,  2, -1, -1,  2,  0,  1,  1,  0,  1,  0,  0,  1, -1,  0,  1,  0,
      0, -1, -1,  0, -1, -3,  1,  1,  3, -1, -1,  3, -2, -1, -2,  2,  1,
      -1,  2,  1,  0,  2,  1, -3,  2,  1, -1,  1,  1,  0,  1,  1,  1,  1,
      1, -3,  1,  1, -2,  1,  1,  0,  0,  1,  1,  0,  1,  2,  0,  1, -3,
      0,  1, -2,  0,  1, -1,  0,  1,  1, -1,  1,  1,  1, -1,  1,  0, -1,
      3, -1,  1,  2, -1,  1,  1, -1,  1,  0, -1,  1, -1,  1, -1, -2,  1,
      -1, -2,  0,  1,  3,  0, -1,  1,  2, -1,  0,  2, -1,  0,  1, -1,  3,
      -2,  1,  2, -2,  1,  1, -2,  1, -2,  2, -1, -1, -1,  1, -2, -1,  1,
      3,  1, -1,  2,  1, -1,  2, -3,  0, -2,  3, -1,  1, -3,  1,  1,  3,
      -1,  0,  3, -1, -1,  3, -1, -1,  2, -1,  2, -3,  1,  0, -2,  1, -1,
      -2,  1,  2,  2, -1,  2, -1,  1, -2, -1,  1, -1, -1,  1,  0, -1,  1,
      2, -2,  1, -1, -2,  1,  0, -2,  1,  1, -2,  1,  0, -3,  1,  1, -3,
      1,  2, -3,  1, -2,  3,  0, -1,  3,  0, -1,  2,  0,  0,  2,  0,  1,
      2,  0,  3, -2,  0,  1,  2,  0,  0,  2,  0, -1,  2,  0, -2,  2,  0,
      1, -2,  0,  0, -2,  0,  2,  2,  0,  0,  3,  0, -1,  3,  0, -2,  3,
      0,  3, -1,  0,  2, -1,  0,  2, -2,  0,  2,  1,  0,  1,  1,  0,  0,
      1,  0, -1,  1,  0,  0, -1,  0, -1, -1,  0, -3,  2,  0, -2, -1,  0,
      3, -1,  0,  2, -1,  0,  1, -1,  0, -1,  0,  0, -2,  0,  0,  1, -1,
      0,  3,  0,  0,  2,  0,  0,  1,  0,  0,  0,  0,  0, -1,  0,  0, -2,
      0,  0, -2,  1,  0, -2,  2, -1,  1,  2, -1,  0,  2, -1, -1,  2, -1,
      0,  3, -1, -1,  3, -1, -2,  3, -1,  2, -3,  0,  1, -3,  0,  1, -2,
      0,  0, -2,  0, -1, -2,  0,  3, -2,  0,  2, -2,  0,  0, -1,  0, -1,
      -1,  0,  1, -1, -1,  0, -1, -1, -1, -1, -1,  3, -1, -1,  2, -1, -1,
      0,  0, -1, -1,  0, -1, -2,  0, -1,  3,  0, -1,  2,  0, -1,  1,  0,
      -1, -1,  1, -1, -2,  1, -1,  2,  1, -1,  1,  1, -1,  0,  1, -1,  2,
      0, -2,  0,  1, -2, -1,  1, -2,  2,  1, -2,  1,  1, -2, -1,  2, -2,
      -2,  2, -2,  2,  2, -2,  1,  2, -2,  0,  2, -2,  0,  3, -2, -1,  3,
      -2,  2, -2, -1,  1, -2, -1,  0, -2, -1,  3, -2, -1, -3,  2,  0, -2,
      2,  0,  0,  1,  0,  1,  1,  0,  2,  1,  0, -3,  1,  0, -2,  1,  0,
      -1,  1,  0,  1,  0,  0,  2,  0,  0
    };

TEST(Op2cEngineTest, TestBuildVelocity) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 2) {
        if(rank==0) std::cout << "Skipping test requiring 2 processes" << std::endl;
        return;
    }
    // op2c's explicit velocity is the beta-projector route (cross-check only;
    // production velocity is the commutator, doc/position_commutator.md), so
    // it needs a pseudopotential file the test data does not ship.
    if (!std::ifstream(rsatb::test::data_path("diamond/C_ONCV_PBE-1.0.upf")).good()) {
        GTEST_SKIP() << "explicit-integral velocity needs diamond/C_ONCV_PBE-1.0.upf";
    }
    
    // 1. Setup AtomicData
    GlobalEnv env;
    AtomicData* ad = nullptr;
    
    if (rank == 0) {
        ad = new AtomicData(
            env.n_atom, env.n_atom*2, env.atom_offset1, env.n_edge, env.n_edge * 2,
            &env.atom_index1, env.atom_type1, edge_index1, env.type_norb, edge_shift_vec1,
            env.cell, env.pos1, MPI_COMM_WORLD
        );
    } else {
        ad = new AtomicData(
            env.n_atom, env.n_atom*2, env.atom_offset2, env.n_edge, env.n_edge * 2,
            &env.atom_index2, env.atom_type2, edge_index2, env.type_norb, edge_shift_vec2,
            env.cell, env.pos2, MPI_COMM_WORLD
        );
    }
    
    // 2. Set up the op2c engine
    Op2cEngine op_const(MPI_COMM_WORLD);
    
    std::string orb_dir = rsatb::test::data_path("diamond/");
    std::vector<std::string> orb_names = {"C_gga_7au_100Ry_2s2p1d.orb", "C_gga_7au_100Ry_2s2p1d.orb"};
    std::string psd_dir = rsatb::test::data_path("diamond/");
    std::vector<std::string> psd_names = {"C_ONCV_PBE-1.0.upf", "C_ONCV_PBE-1.0.upf"};
    
    op_const.init(2, 1, false, orb_dir, orb_names, psd_dir, psd_names, "op_const.log");
    
    // 3. Build Velocity
    if(rank==0) std::cout << "Allocating 3-body graph..." << std::endl;
    std::vector<double> r_max_left={7.5, 7.5};
    std::vector<double> r_max_right={1.45, 1.45};
    DistGraph* g3 = ad->get_graph3b(r_max_left, r_max_right);
    if(rank==0) std::cout << "Allocating 3-body graph DONE." << std::endl;
    
    BlockSpMat<std::complex<double>> S(ad->graph);
    BlockSpMat<std::complex<double>> Vx3(g3);
    BlockSpMat<std::complex<double>> Vy3(g3);
    BlockSpMat<std::complex<double>> Vz3(g3);
    
    if(rank==0) std::cout << "Building velocity..." << std::endl;
    if(rank==0) std::cout << "Building velocity..." << std::endl;
    auto img_S = op_const.build_overlap_velocity(&Vx3, &Vy3, &Vz3, *ad, 0.0, 0.0, 0.0);
    if(rank==0) std::cout << "Building velocity DONE." << std::endl;
    if(rank==0) std::cout << "Building velocity DONE." << std::endl;
    
    // Verify
    // Check if matrices have non-zero elements
    // We can check local values
    double norm_x = 0.0;
    for (size_t i = 0; i < Vx3.local_block_nnz(); ++i) {
        const std::complex<double>* data = Vx3.block_data(static_cast<int>(i));
        const size_t n_elem = Vx3.block_size_elements(static_cast<int>(i));
        for (size_t j = 0; j < n_elem; ++j) {
            norm_x += std::abs(data[j]);
        }
    }
    
    double global_norm_x;
    MPI_Allreduce(&norm_x, &global_norm_x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Global Norm Vx: " << global_norm_x << std::endl;
        EXPECT_GT(global_norm_x, 0.0);
    }
    
    delete ad;
    delete g3;
}

TEST(Op2cEngineTest, TestBuildOverlap) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 2) return;

    GlobalEnv env;
    AtomicData* ad = nullptr;
    
    if (rank == 0) {
        ad = new AtomicData(
            env.n_atom, env.n_atom*2, env.atom_offset1, env.n_edge, env.n_edge * 2,
            &env.atom_index1, env.atom_type1, edge_index1, env.type_norb, edge_shift_vec1,
            env.cell, env.pos1, MPI_COMM_WORLD
        );
    } else {
        ad = new AtomicData(
            env.n_atom, env.n_atom*2, env.atom_offset2, env.n_edge, env.n_edge * 2,
            &env.atom_index2, env.atom_type2, edge_index2, env.type_norb, edge_shift_vec2,
            env.cell, env.pos2, MPI_COMM_WORLD
        );
    }
    
    Op2cEngine op_const(MPI_COMM_WORLD);
    
    std::string orb_dir = rsatb::test::data_path("diamond/");
    std::vector<std::string> orb_names = {"C_gga_7au_100Ry_2s2p1d.orb", "C_gga_7au_100Ry_2s2p1d.orb"};
    
    op_const.init(2, 1, false, orb_dir, orb_names, "", {}, "op_const_overlap.log");
    

    
    if(rank==0) std::cout << "Building overlap..." << std::endl;
    if(rank==0) std::cout << "Building overlap..." << std::endl;
    auto img_S = op_const.build_overlap(*ad);
    if(rank==0) std::cout << "Building overlap DONE." << std::endl;
    
    auto* S_ptr = img_S->sample_k({0,0,0}, PhaseConvention::R_AND_POSITION);
    BlockSpMat<std::complex<double>>& S = *S_ptr;
    
    double norm_s = 0.0;
    for (size_t i = 0; i < S.local_block_nnz(); ++i) {
        const std::complex<double>* data = S.block_data(static_cast<int>(i));
        const size_t n_elem = S.block_size_elements(static_cast<int>(i));
        for (size_t j = 0; j < n_elem; ++j) {
            norm_s += std::abs(data[j]);
        }
    }
    
    double global_norm_s;
    MPI_Allreduce(&norm_s, &global_norm_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Global Norm S: " << global_norm_s << std::endl;
        EXPECT_GT(global_norm_s, 0.0);
    }
    
    delete S_ptr;
    delete ad;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}

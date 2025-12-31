#include "hr2hk.hpp"
#include "solver.hpp"
#include <gtest/gtest.h>
#include <complex>

using namespace rsatb;
using namespace rsatb::io;
using namespace rsatb::backend;

// Reusing GlobalEnv from test.cpp for consistent setup
class GlobalEnv : public ::testing::Environment {
private:
    rsatb::Solver *solver_;
    int rank, size;
    int n_atom=1, n_edge=292;

    int atom_type1[1] = {0};
    int atom_type2[1] = {1};
    int type_norb[2] = {13, 13};
    
    double cell[9] = {2.52685 , 0., 0., 1.263425, 2.188316, 0., 1.263425, 0.729439, 2.063165};
    std::string HFile = "test/data/diamond/C2.0/hamiltonians.h5";
    std::string SFile = "test/data/diamond/C2.0/overlaps.h5";

    int atom_offset1=0;
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
    double pos1[3] = {2.52685   , 1.4588775 , 1.0315825};
    int atom_index1 = 0;

    int atom_offset2=1;
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
    double pos2[3] = {3.790275  , 2.18831625, 1.54737375};
    int atom_index2 = 1;
public:
  void SetUp() override {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 2) {
      throw std::runtime_error("The test is used for two process!");
    };

    if (rank == 0){
      solver_ = new rsatb::Solver(
          n_atom, n_atom*2, atom_offset1, n_edge, n_edge * 2, 
          &atom_index1, atom_type1, edge_index1, type_norb, edge_shift_vec1,
          cell, pos1, MPI_COMM_WORLD
      );
    } else {
      solver_ = new rsatb::Solver(
          n_atom, n_atom*2, atom_offset2, n_edge, n_edge * 2, 
          &atom_index2, atom_type2, edge_index2, type_norb, edge_shift_vec2,
          cell, pos2, MPI_COMM_WORLD
      );
    };

    std::cout << "Solver init SUCCESS!\n";
  }
  void TearDown() override {
    delete solver_;
  }

  rsatb::Solver& solver() {return *solver_; }
};

static GlobalEnv* g_env = nullptr;

GlobalEnv& GetGlobalEnv() {
  if (!g_env) throw std::runtime_error("GlobalEnv not installed");
  return *g_env;
}

TEST(Hr2hkTest, TestCalcPhase) {
    std::complex<double> phase;
    cal_phase(0, 0, 0, 0.5, 0.5, 0.5, &phase);
    ASSERT_NEAR(phase.real(), 1.0, 1e-10);
    ASSERT_NEAR(phase.imag(), 0.0, 1e-10);

    cal_phase(1, 1, 1, 0.5, 0.5, 0.5, &phase);
    ASSERT_NEAR(phase.real(), -1.0, 1e-10);
    ASSERT_NEAR(phase.imag(), 0.0, 1e-10);
}

TEST(Hr2hkTest, TestSetMatrixR2K) {
    GlobalEnv& env = GetGlobalEnv();
    auto& solver = env.solver();
    auto& dm = solver.dm_atomic();
    
    // Test parameters
    double Kx = 0.0, Ky = 0.0, Kz = 0.0;
    std::string file_orb0[2] = {"C_gga_7au_100Ry_2s2p1d.orb", "C_gga_7au_100Ry_2s2p1d.orb"};
    std::string root = "/home/zhanghao/softwares/rsatb/test/data/diamond/";
    
    // 1. Test Overlap Matrix Construction
    Mat S;
    DMCreateMatrix(dm.dm(), &S);
    MatConvert(S, "aij", MAT_INPLACE_MATRIX, &S);
    
    setMatrixR2K(S, dm, file_orb0, root, Kx, Ky, Kz);
    
    // Verify properties
    MatSetOption(S, MAT_HERMITIAN, PETSC_TRUE);
    
    PetscInt M, N;
    MatGetSize(S, &M, &N);
    EXPECT_GT(M, 0);
    EXPECT_GT(N, 0);

    MatDestroy(&S);
    DMCreateMatrix(dm.dm(), &S);
    
    // Check Hermiticity
    // PetscBool isHermitian;
    // MatIsHermitian(S, 1e-8, &isHermitian);
    // EXPECT_TRUE(isHermitian);

    // 2. Test Velocity Operator Construction
    Mat Vx=NULL, Vy=NULL, Vz=NULL;
    std::vector<double> r_max_left={7.5, 7.5}; // cutoff from orbitals
    std::vector<double> r_max_right{1.45, 1.45}; // cutoff from pseudo potentials

    // std::cout << "dm.allocate_three_body_hopping_matrix\n";
    dm.allocate_three_body_hopping_matrix(&Vx, r_max_left, r_max_right);
    dm.allocate_three_body_hopping_matrix(&Vy, r_max_left, r_max_right);
    dm.allocate_three_body_hopping_matrix(&Vz, r_max_left, r_max_right);
    // std::cout << "dm.allocate_three_body_hopping_matrix END\n";
    // DMCreateMatrix(dm.dm(), &Vx);
    // DMCreateMatrix(dm.dm(), &Vy);
    // DMCreateMatrix(dm.dm(), &Vz);
    // MatConvert(Vx, "aij", MAT_INPLACE_MATRIX, &Vx);
    // MatConvert(Vy, "aij", MAT_INPLACE_MATRIX, &Vy);
    // MatConvert(Vz, "aij", MAT_INPLACE_MATRIX, &Vz);
    
    // We need to pass empty vectors for orb_names and psd_names as per the signature
    // void setMatrixR2K(Mat S, rsatb::backend::DMAtomic &dm_atomic, const double* cell, int ntype, 
    //             const std::vector<std::string> orb_names, const std::string& orb_dir, const std::vector<std::string> psd_names, const std::string& psd_dir, 
    //             double Kx, double Ky, double Kz, Mat Vx=nullptr, Mat Vy=nullptr, Mat Vz=nullptr, const std::string& log_file="");
    
    std::vector<std::string> orb_names = {file_orb0[0], file_orb0[1]};
    std::vector<std::string> psd_names = {"C_ONCV_PBE-1.0.upf", "C_ONCV_PBE-1.0.upf"};
    // Looking at hr2hk.cpp, psd_names are used for reading pseudo potentials if needed.
    // For diamond test, we might not need psd_names if we are just testing velocity op which depends on orbitals.
    // However, let's check the implementation of setMatrixR2K in hr2hk.cpp to be sure.
    // Assuming for now empty is  or we can provide dummy if it crashes.
    
    // Re-using S for the first argument as it is also computed/updated
    // std::cout << "rank: " << dm.rank() << ", " << "Before filling the V matrix\n";
    setMatrixR2K(S, dm, orb_names, root, psd_names, root, Kx, Ky, Kz, Vx, Vy, Vz,"log.txt");
    // std::cout << "rank: " << dm.rank() << ", " << "After filling the V matrix\n" << std::endl;
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "./S.mtx", &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATRIXMARKET);
    MatView(S, viewer);
    PetscViewerDestroy(&viewer);

    PetscViewer viewer2;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "./Vx.mtx", &viewer2);
    PetscViewerPushFormat(viewer2, PETSC_VIEWER_ASCII_MATRIXMARKET);
    MatView(Vx, viewer2);
    PetscViewerDestroy(&viewer2);

    PetscViewer viewer3;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "./Vy.mtx", &viewer3);
    PetscViewerPushFormat(viewer3, PETSC_VIEWER_ASCII_MATRIXMARKET);
    MatView(Vy, viewer3);
    PetscViewerDestroy(&viewer3);

    PetscViewer viewer4;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "./Vz.mtx", &viewer4);
    PetscViewerPushFormat(viewer4, PETSC_VIEWER_ASCII_MATRIXMARKET);
    MatView(Vz, viewer4);
    PetscViewerDestroy(&viewer4);

    // Verify Velocity Operators are Hermitian (Velocity operator is Hermitian? p = -ih bar grad. In k-space, it is Hermitian)
    // MatIsHermitian(Vx, 1e-8, &isHermitian);
    // EXPECT_TRUE(isHermitian);
    // MatIsHermitian(Vy, 1e-8, &isHermitian);
    // EXPECT_TRUE(isHermitian);
    // MatIsHermitian(Vz, 1e-8, &isHermitian);
    // EXPECT_TRUE(isHermitian);

    MatDestroy(&S);
    MatDestroy(&Vx);
    MatDestroy(&Vy);
    MatDestroy(&Vz);
}

int main(int argc, char** argv)
{   
    MPI_Init(&argc, &argv);
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    SlepcInitialize(&argc,&argv,NULL,NULL);
    c_dbcsr_init_lib(MPI_COMM_WORLD, nullptr);
    ::testing::InitGoogleTest(&argc, argv);
    g_env = static_cast<GlobalEnv*>(
      ::testing::AddGlobalTestEnvironment(new GlobalEnv));
    int result = RUN_ALL_TESTS();
    c_dbcsr_finalize_lib();
    SlepcFinalize();
    PetscFinalize();
    MPI_Finalize();
    return result;
}

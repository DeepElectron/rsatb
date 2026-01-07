#include "hr2hk.hpp"
#include "solver.hpp"
#include "calc/isqrtm.hpp"
#include <gtest/gtest.h>

using namespace rsatb;
using namespace rsatb::io;
using namespace rsatb::backend;

class GlobalEnv : public ::testing::Environment {
private:
    rsatb::Solver *solver_;
    int rank, size;
    int n_atom=1, n_edge=292;

    int atom_type1[1] = {0};
    int atom_type2[1] = {1};
    int type_norb[2] = {13, 13};
    
    double cell[9] = {2.52685 , 0., 0., 1.263425, 2.188316, 0., 1.263425, 0.729439, 2.063165};
    std::string HFile = "../../test/data/diamond/C2.0/hamiltonians.h5";
    std::string SFile = "../../test/data/diamond/C2.0/overlaps.h5";

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
    // e.g., PetscInitialize(...)
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

    solver_->setup_sample(100, "gauss");
    std::cout << "Solver setup_sample SUCCESS!\n";
    solver_->setup_K(0.3,-0.72,0.133);
    std::cout << "Solver setup_K SUCCESS!\n";
    solver_->setup_S(SFile);
    std::cout << "Solver setup_S(file) SUCCESS!\n";

    std::string file_orb0[2] = {"C_gga_7au_100Ry_2s2p1d.orb", "C_gga_7au_100Ry_2s2p1d.orb"};
    std::string root = "../../test/data/diamond/";
    solver_->setup_S(file_orb0, root);
    std::cout << "Solver setup_S(2cint) SUCCESS!\n";

    solver_->Ssqrtm(1e-7, 1e-6, 20, true);
    std::cout << "Solver Sqrtm SUCCESS!\n";
    solver_->setup_H_h5(HFile);
    std::cout << "Solver Setup_H SUCCESS!\n";
    // solver_->setup_R(file_orb0, root);
  }
  void TearDown() override {
    // e.g., PetscFinalize()
    delete solver_;
  }

  rsatb::Solver& solver() {return *solver_; }
};

static GlobalEnv* g_env = nullptr;

GlobalEnv& GetGlobalEnv() {
  if (!g_env) throw std::runtime_error("GlobalEnv not installed");
  return *g_env;
}

// TEST(petscTest, TestCalcPhase) {
//     std::complex<double> phase;
//     cal_phase(0, 0, 0, 0.5, 0.5, 0.5, &phase);
//     ASSERT_NEAR(phase.real(), 1.0, 1e-10);
//     ASSERT_NEAR(phase.imag(), 0.0, 1e-10);

//     cal_phase(1, 1, 1, 0.5, 0.5, 0.5, &phase);
//     ASSERT_NEAR(phase.real(), -1.0, 1e-10);
//     ASSERT_NEAR(phase.imag(),0.0, 1e-10);
// }

// TEST(petscTest, TestSetBlockR2K) {
//     int n_node=4, n_edge=7;
//     int edge_index[] = {0,1,1,0,2,3,3,2,0,3,3,0,0,2,2,0};
//     int atom_type[] = {0,0,0,0};
//     int type_weight[] = {1};
//     int edge_shift_vec[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
//     double pos[] = {0.0,0.5,0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.0,0.0,0.5};
//     DM dm;
//     MPI_Comm comm = MPI_COMM_WORLD;
//     int rank;
//     MPI_Comm_rank(comm, &rank);
//     rsatb::backend::get_dm_atomic(
//       n_node, n_edge, edge_index, atom_type, type_weight, edge_shift_vec, pos, &dm, comm
//     );
//     Mat mat;
//     DMCreateMatrix(dm, &mat);
//     MatView(mat, PETSC_VIEWER_STDOUT_WORLD);
//     double v[] = {1.0};
//     PetscInt p;
//     PetscBool ghost=PETSC_TRUE;
//     DMNetworkGetVertexRange(dm, &p, NULL);
//     DMNetworkIsGhostVertex(dm,p,&ghost);
//     while (ghost)
//     {
//         p += 1;
//         DMNetworkIsGhostVertex(dm,p,&ghost);
        
//     }
//     std::cout << "vertice " << p << " on rank " << rank << "\n";
//     rsatb::SetBlockR2K(mat, dm, p, p, 1, 1, 0,0,0, 0.0,0.0,0.0, v);
//     MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
//     MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);
//     MatView(mat, PETSC_VIEWER_STDOUT_WORLD);
//     ViewDMNetworkPartition(dm,false);
// }

// TEST(petscTest, TestSetMatrixR2K) {
//     int n_node=3, n_edge=6;
//     int edge_index[] = {0, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 2};
//     int atom_type[] = {0,0,0};
//     int type_weight[] = {4};
//     int edge_shift_vec[] = {0,  0,  0,  0,  0,  1,  0,  0,  -1,  0,  0,  -1,  0,  0, 0,  0,  0,  1};
//     double pos[] = {0.0000, 0.0000, 3.2000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.6000};
//     std::string path = "../../test/data/C_chain_3_H.h5";
//     rsatb::io::HDF5Handler hd(path);
//     DM dm;
//     MPI_Comm comm = MPI_COMM_WORLD;
//     int rank;
//     MPI_Comm_rank(comm, &rank);
//     rsatb::backend::get_dm_atomic(
//       n_node, n_edge, edge_index, atom_type, type_weight, edge_shift_vec, pos, &dm, comm
//     );
//     Mat mat, mat2, mat3;
//     DMCreateMatrix(dm, &mat);
//     setMatrixR2K(mat, dm, &hd, 0., 0., 0.);
//     MatDuplicate(mat, MAT_COPY_VALUES, &mat2);
//     MatConjugate(mat2);
//     MatTranspose(mat2, MAT_INITIAL_MATRIX, &mat3);
//     MatAXPY(mat, -1.0, mat3, SAME_NONZERO_PATTERN);
//     MatView(mat, PETSC_VIEWER_STDOUT_WORLD);
//     rsatb::backend::ViewDMNetworkPartition(dm, false);
// }

// TEST(petscTest, TestSolverInit) {
    
//     // PetscScalar va;
//     // std::cout << "######## Analytical: "  << std::endl;
//     // for (PetscInt i=0; i<7; ++i) {
//     //     for (PetscInt j=0; j<7; ++j) {
//     //         MatGetValue(solver.S(), i, j, &va);
//     //         std::cout << va.real() << "\t";
//     //     }
//     //     std::cout << "\n";
//     // }

//     // std::cout << "######## predicted: "  << std::endl;
//     // for (PetscInt i=0; i<7; ++i) {
//     //     for (PetscInt j=0; j<7; ++j) {
//     //         MatGetValue(Sh5copy, i, j, &va);
//     //         std::cout << va.real() << "\t";
//     //     }
//     //     std::cout << "\n";
//     // }

//     MatTranspose(solver.S(), MAT_INITIAL_MATRIX, &SorbT);
//     MatAXPY(SorbT, -1., solver.S(), SAME_NONZERO_PATTERN);
//     MatAXPY(Sh5copy, -1., solver.S(), SAME_NONZERO_PATTERN);

    
//     PetscReal nSd, nS, nST;
//     MatNorm(solver.S(), NORM_FROBENIUS, &nS);
//     MatNorm(Sh5copy, NORM_FROBENIUS, &nSd);
//     MatNorm(SorbT, NORM_FROBENIUS, &nST);
//     std::cout << "The norm of the overlap: " << nS << std::endl;
//     std::cout << "The norm of the overlap deriv: " << nSd << std::endl;
//     std::cout << "The norm of the overlap transpose of analytical one: " << nST << std::endl;

//     solver.setup_H(HFile);
//     std::cout << "Solver setup_H SUCCESS!\n";   // Estimated Emax : (47.4771,0)  Emin : (-105.726,0)

//     solver.setup_R(file_orb0, root);

//     // std::cout << "######## computed Rx: "  << std::endl;
//     // for (PetscInt i=21; i<28; ++i) {
//     //     for (PetscInt j=21; j<28; ++j) {
//     //         MatGetValue(solver.Rx(), i, j, &va);
//     //         std::cout << va.real() << "\t";
//     //     }
//     //     std::cout << "\n";
//     // }
//     // std::cout << "######## computed Ry: "  << std::endl;
//     // for (PetscInt i=21; i<28; ++i) {
//     //     for (PetscInt j=21; j<28; ++j) {
//     //         MatGetValue(solver.Ry(), i, j, &va);
//     //         std::cout << va.real() << "\t";
//     //     }
//     //     std::cout << "\n";
//     // }
//     // std::cout << "######## computed Rz: "  << std::endl;
//     // for (PetscInt i=21; i<28; ++i) {
//     //     for (PetscInt j=21; j<28; ++j) {
//     //         MatGetValue(solver.Rz(), i, j, &va);
//     //         std::cout << va.real() << "\t";
//     //     }
//     //     std::cout << "\n";
//     // }
// }

// TEST(petscTest, TestSolverAC) {

//     // init finish, test start
//     GlobalEnv& env = GetGlobalEnv();

//     int nt = 128;

//     double ac[6*(nt+1)];
//     double omegas[nt+1];
    
//     env.solver().compute_ac(nt, 500, 300.0, -50.0, ac, omegas);

//     std::cout << "[";
//     for(int i=0; i<6*(nt+1); ++i){
//         std::cout << ac[i] << ", ";
//     }
//     std::cout << "]";
//     std::cout << std::endl;
// }

TEST(petscTest, TestSolverDOS) {

    // init finish, test start
    GlobalEnv& env = GetGlobalEnv();

    int nt = 128, nproj=1;

    double dos[2*nt] = {0};
    double omega[2*nt] = {0};
    std::unique_ptr<double[]> pdos;
    
    env.solver().compute_dos(nt, "none", nullptr, nullptr, dos, pdos, omega, true);

    if (env.solver().rank == 0){
      std::cout << "[";
      for(int i=0; i<(2*nt); ++i){
          std::cout << dos[i] << ", ";
      }
      std::cout << "]";
      std::cout << std::endl;
    }
}

int main(int argc, char** argv)
{   
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
    return result;
}


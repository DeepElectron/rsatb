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
    int n_atom=27, n_edge=324;
    int edge_index[648] = {0,  2,  0, 11,  0,  9,  0, 21,  0, 18,  0, 19,  0, 15,  0,  7,  0,
        6,  0,  5,  0,  1,  0,  3,  1,  8,  1, 20,  1,  2,  1,  7,  1, 16,
        1, 10,  1,  9,  1, 22,  1, 19,  1,  3,  1,  4,  2,  4,  2, 11,  2,
       18,  2,  8,  2, 17,  2,  5,  2, 20,  2, 23,  2,  6,  2, 10,  3,  9,
        3,  5,  3,  8,  3,  6,  3, 14,  3, 24,  3, 21,  3, 22,  3, 12,  3,
        4,  4, 23,  4,  5,  4, 22,  4,  6,  4,  7,  4, 13,  4, 12,  4, 10,
        4, 25,  5, 26,  5,  7,  5,  8,  5, 11,  5, 14,  5, 23,  5, 13,  5,
       21,  6, 25,  6, 12,  6, 24,  6, 15,  6,  7,  6,  8,  6, 17,  6, 18,
        7, 26,  7,  8,  7, 15,  7, 16,  7, 13,  7, 25,  7, 19,  8, 24,  8,
       14,  8, 16,  8, 26,  8, 17,  8, 20,  9, 10,  9, 12,  9, 14,  9, 11,
        9, 24,  9, 18,  9, 15,  9, 20,  9, 16, 10, 12, 10, 13, 10, 18, 10,
       19, 10, 16, 10, 17, 10, 25, 10, 11, 11, 17, 11, 19, 11, 26, 11, 20,
       11, 13, 11, 14, 11, 15, 12, 23, 12, 13, 12, 14, 12, 17, 12, 15, 12,
       18, 12, 21, 13, 16, 13, 14, 13, 19, 13, 15, 13, 22, 13, 21, 14, 16,
       14, 22, 14, 17, 14, 20, 14, 23, 15, 16, 15, 17, 15, 26, 15, 21, 15,
       24, 16, 22, 16, 24, 16, 25, 16, 17, 17, 26, 17, 23, 17, 25, 18, 24,
       18, 25, 18, 23, 18, 19, 18, 21, 18, 20, 19, 25, 19, 21, 19, 22, 19,
       26, 19, 20, 20, 24, 20, 22, 20, 26, 20, 23, 21, 23, 21, 24, 21, 26,
       21, 22, 22, 23, 22, 24, 22, 25, 23, 25, 23, 26, 24, 25, 24, 26, 25,
       26,  2,  0, 11,  0,  9,  0, 21,  0, 18,  0, 19,  0, 15,  0,  7,  0,
        6,  0,  5,  0,  1,  0,  3,  0,  8,  1, 20,  1,  2,  1,  7,  1, 16,
        1, 10,  1,  9,  1, 22,  1, 19,  1,  3,  1,  4,  1,  4,  2, 11,  2,
       18,  2,  8,  2, 17,  2,  5,  2, 20,  2, 23,  2,  6,  2, 10,  2,  9,
        3,  5,  3,  8,  3,  6,  3, 14,  3, 24,  3, 21,  3, 22,  3, 12,  3,
        4,  3, 23,  4,  5,  4, 22,  4,  6,  4,  7,  4, 13,  4, 12,  4, 10,
        4, 25,  4, 26,  5,  7,  5,  8,  5, 11,  5, 14,  5, 23,  5, 13,  5,
       21,  5, 25,  6, 12,  6, 24,  6, 15,  6,  7,  6,  8,  6, 17,  6, 18,
        6, 26,  7,  8,  7, 15,  7, 16,  7, 13,  7, 25,  7, 19,  7, 24,  8,
       14,  8, 16,  8, 26,  8, 17,  8, 20,  8, 10,  9, 12,  9, 14,  9, 11,
        9, 24,  9, 18,  9, 15,  9, 20,  9, 16,  9, 12, 10, 13, 10, 18, 10,
       19, 10, 16, 10, 17, 10, 25, 10, 11, 10, 17, 11, 19, 11, 26, 11, 20,
       11, 13, 11, 14, 11, 15, 11, 23, 12, 13, 12, 14, 12, 17, 12, 15, 12,
       18, 12, 21, 12, 16, 13, 14, 13, 19, 13, 15, 13, 22, 13, 21, 13, 16,
       14, 22, 14, 17, 14, 20, 14, 23, 14, 16, 15, 17, 15, 26, 15, 21, 15,
       24, 15, 22, 16, 24, 16, 25, 16, 17, 16, 26, 17, 23, 17, 25, 17, 24,
       18, 25, 18, 23, 18, 19, 18, 21, 18, 20, 18, 25, 19, 21, 19, 22, 19,
       26, 19, 20, 19, 24, 20, 22, 20, 26, 20, 23, 20, 23, 21, 24, 21, 26,
       21, 22, 21, 23, 22, 24, 22, 25, 22, 25, 23, 26, 23, 25, 24, 26, 24,
       26, 25};
    int atom_type[27] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int type_norb[1] = {21};
    int edge_shift_vec[972] = {
        0,  0, -1,  0,  0, -1,  0,  0,  0, -1,  0,  0, -1,  0,  0, -1,  0,
        0,  0, -1,  0,  0, -1,  0,  0, -1,  0,  0,  0, -1,  0,  0,  0,  0,
        0,  0,  0, -1,  0, -1,  0,  0,  0,  0,  0,  0, -1,  0,  0, -1,  0,
        0,  0,  0,  0,  0,  0, -1,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0, -1,  0,  1,  0, -1,  0,  0, -1,  0,  0,
        0,  0, -1,  0,  0, -1,  0,  0,  0, -1,  1,  0,  0,  0,  0,  0,  0,
        0,  0, -1,  0,  0, -1,  0,  0,  0,  0,  0, -1, -1,  0,  0, -1,  0,
        0, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
       -1,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0, -1,  0,  0,  0,  0,  0, -1,  0,  1, -1,  0,  0,  0,  0,  0, -1,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0, -1, -1,  1,  0,
       -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,
        0, -1,  1,  0, -1,  0,  1,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,
        0,  0, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0, -1,
        0, -1,  0,  0,  0,  0,  0, -1,  0,  0,  0, -1,  0, -1,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0, -1,  0,  0,
       -1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0, -1,  0,  0,  0,  0,  0,
       -1,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
       -1,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,
        0, -1,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0, -1,
        0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1,  1,  0,
        0,  0,  0, -1,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,
        0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  0,  1,  0,  0,  1,  0,
        0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,
        0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,
        0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        1,  0, -1,  0,  1,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  1,  0,
        0,  0,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,
        0,  0,  0,  0,  1,  1,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,
        0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,
        1,  0, -1,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  1,  0,  0,  1,  1, -1,  0,  1,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1, -1,  0,  1,  0, -1,
        0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1, -1,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,
        1,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  1,
        0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        1, -1,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  1,  0,
        0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,
        0,  1,  0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  0,  0,  0,
        0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,
        0,  0,  0};
    double pos[81] = { 0.        ,  0.        ,  0.        ,  2.70554805,  1.56204889,
        4.41814146,  5.4110961 ,  3.12409779,  8.83628293,  2.70554805,
        4.68614668,  0.        ,  5.41109609,  6.24819557,  4.41814146,
        8.11664415,  7.81024447,  8.83628293,  5.4110961 ,  9.37229337,
        0.        ,  8.11664415, 10.93434227,  4.41814146, 10.8221922 ,
       12.49639116,  8.83628293,  5.41109609,  0.        ,  0.        ,
        8.11664414,  1.56204889,  4.41814146, 10.8221922 ,  3.12409779,
        8.83628293,  8.11664414,  4.68614668,  0.        , 10.82219219,
        6.24819557,  4.41814146, 13.52774024,  7.81024447,  8.83628293,
       10.8221922 ,  9.37229337,  0.        , 13.52774024, 10.93434227,
        4.41814146, 16.2332883 , 12.49639116,  8.83628293, 10.8221922 ,
        0.        ,  0.        , 13.52774025,  1.56204889,  4.41814146,
       16.23328831,  3.12409779,  8.83628293, 13.52774025,  4.68614668,
        0.        , 16.2332883 ,  6.24819557,  4.41814146, 18.93883635,
        7.81024447,  8.83628293, 16.23328831,  9.37229337,  0.        ,
       18.93883635, 10.93434227,  4.41814146, 21.64438441, 12.49639116,
        8.83628293};
    
    double cell[9] = {16.2332878113, 0.0, 0.0, 8.1166439056, 14.0584396315, 0.0, 8.1166439056, 4.6861465438, 13.2544239951};
    std::string HFile = "../../test/data/Al/H_Al333.h5";
    std::string SFile = "../../test/data/Al/S_Al333.h5";
public:
  void SetUp() override {
    // e.g., PetscInitialize(...)
    solver_ = new rsatb::Solver(
        n_atom, n_edge, 
        atom_type, edge_index, type_norb, edge_shift_vec,
        cell, pos, MPI_COMM_WORLD
    );

    std::cout << "Solver init SUCCESS!\n";

    solver_->setup_graph();
    std::cout << "Solver setup_graph SUCCESS!\n";
    solver_->setup_sample(10, "gauss");
    std::cout << "Solver setup_sample SUCCESS!\n";
    solver_->setup_K(0.5,0.5,0.5);
    std::cout << "Solver setup_K SUCCESS!\n";
    // solver_->setup_S(SFile);
    // std::cout << "Solver setup_S SUCCESS!\n";

    std::string file_orb0[1] = {"Al_gga_7au_100Ry_4s4p1d.orb"};
    std::string root = "../../test/data/Al/";
    solver_->setup_S(file_orb0, root);
    // solver_->setup_H(HFile);
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

TEST(calcTest, Testisqrt){
    GlobalEnv& env = GetGlobalEnv();
    dbcsr_matrix dmat;
    Mat mat_=nullptr;
    // DMat2DBCSR(env.solver().S(), env.solver().dm(), &dmat);
    // c_dbcsr_release(&dmat);
    
    isqrtm(env.solver().S(), env.solver().dm(), 1e-7, 1e-6, 20, 0.05, &mat_);
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


// #include <gtest/gtest.h>
// #include "petsc.hpp"
// #include "dbcsr.hpp"
// #include "dbcsr_capi.h"
// #include <petsc.h>
// #include <mpi.h>
// #include <iostream>

// using namespace rsatb::backend;

// TEST(petscTest, TestGetDMAtomic) {
//     int n_node=2, n_edge=2;
//     int edge_index[] = {0,1,1,0};
//     int atom_type[] = {0,0};
//     int type_weight[] = {1};
//     int edge_shift_vec[] = {0,0,0,0,0,0};
//     double pos[] = {0.5,0.5,0.5,0.0,0.0,0.0};
//     DM dm;
//     MPI_Comm comm = MPI_COMM_WORLD;
//     rsatb::backend::get_dm_atomic(
//       n_node, n_edge, edge_index, atom_type, type_weight, edge_shift_vec, pos, &dm, comm
//     );
// }

// TEST(petscTest, TestGetDMMatrix) {
//     int n_node=4, n_edge=6;
//     int edge_index[] = {0,1,1,0,2,3,3,2,0,2,2,0};
//     int atom_type[] = {0,0,0,0};
//     int type_weight[] = {1};
//     int edge_shift_vec[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
//     double pos[] = {0.0,0.5,0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.0,0.0,0.5};
//     DM dm;
//     MPI_Comm comm = MPI_COMM_WORLD;
//     rsatb::backend::get_dm_atomic(
//       n_node, n_edge, edge_index, atom_type, type_weight, edge_shift_vec, pos, &dm, comm
//     );
//     Mat mat;
//     DMCreateMatrix(dm, &mat);
//     PetscInt first_row, last_row, nghosts;
//     const PetscInt *ghost_cols;
//     MatGetOwnershipRange(mat, &first_row, &last_row);
//     int size;
//     MPI_Comm_size(comm, &size);
//     if(size>1){
//       MatGetGhosts(mat, &nghosts, &ghost_cols);
//       std::cout << first_row << " " << last_row << " " << nghosts << " " << ghost_cols[0] << "\n";
//     }else{
//       MatGetGhosts(mat, &nghosts, &ghost_cols);
//       std::cout << first_row << " " << last_row << " " << nghosts << "\n";
//     }
    
    
//     MatView(mat, PETSC_VIEWER_STDOUT_WORLD);
// }

// TEST(petscTest, TestGetAttr) {
//     int n_node=4, n_edge=6;
//     int edge_index[] = {0,0,1,0,2,3,3,2,0,3,3,0};
//     int atom_type[] = {0,0,0,0};
//     int type_weight[] = {1};
//     int edge_shift_vec[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
//     double pos[] = {0.0,0.5,0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.0,0.0,0.5};
//     DM dm;
//     MPI_Comm comm = MPI_COMM_WORLD;
//     int rank;
//     MPI_Comm_rank(comm, &rank);
//     rsatb::backend::get_dm_atomic(
//       n_node, n_edge, edge_index, atom_type, type_weight, edge_shift_vec, pos, &dm, comm
//     );

//     rsatb::backend::AtomData node;

//     int nvar;
//     PetscInt estart, eend, nstart, nend;
//     DMNetworkGetEdgeRange(dm, &estart, &eend);
//     DMNetworkGetVertexRange(dm, &nstart, &nend);
//     std::cout << "Vertex Range:\t" << nstart << "\t" << nend << " on rank\t" << rank << " \n";
//     std::cout << "Edge Range:\t" << estart << "\t" << eend << " on rank\t" << rank << " \n";
//     rsatb::backend::get_attr_from_dm<rsatb::backend::AtomData>(dm, "atom", nstart, &node, &nvar);
//     ASSERT_EQ(node.x, 0.0);
//     ASSERT_EQ(node.R[0], 0);
//     std::cout << "node 0 " << node.R[0] << node.R[1] << node.R[2] << "\n";

    
//     rsatb::backend::BondData edge;
//     rsatb::backend::get_attr_from_dm<rsatb::backend::BondData>(dm, "edge", estart, &edge, &nvar);
//     ASSERT_EQ(edge.Rx, 0);
// }


// TEST(petscTest, TestGetIndex) {
//     int n_node=4, n_edge=4;
//     int edge_index[] = {0,1,1,0,2,3,3,2};
//     int atom_type[] = {0,0,0,0};
//     int type_weight[] = {1};
//     int edge_shift_vec[] = {0,0,0,0,0,0,0,0,0,0,0,0};
//     double pos[] = {0.0,0.5,0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.0,0.0,0.5};
//     DM dm;
//     MPI_Comm comm = MPI_COMM_WORLD;
//     rsatb::backend::get_dm_atomic(
//       n_node, n_edge, edge_index, atom_type, type_weight, edge_shift_vec, pos, &dm, comm
//     );

//     int i, j;
//     PetscInt estart, eend;
//     DMNetworkGetEdgeRange(dm, &estart, &eend);
//     rsatb::backend::get_edge_index_from_dm(dm, estart+0, &i, &j);
// }

// TEST(petscTest, TestSetupISAtomic) {
//     int n_node=4, n_edge=6;
//     int edge_index[] = {0,1,1,0,2,3,3,2,0,3,3,0};
//     int atom_type[] = {0,0,1,1};
//     int type_weight[] = {3,3};
//     int edge_shift_vec[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
//     double pos[] = {0.0,0.5,0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.0,0.0,0.5};
//     DM dm;
//     MPI_Comm comm = MPI_COMM_WORLD;
//     rsatb::backend::get_dm_atomic(
//       n_node, n_edge, edge_index, atom_type, type_weight, edge_shift_vec, pos, &dm, comm
//     );

//     IS isSrc, isDst;

//     setup_is_atomic(dm, &isSrc, &isDst);
// }

// TEST(petscTest, TestSetupISOrbital) {
//     int n_node=4, n_edge=6;
//     int edge_index[] = {0,1,1,0,2,3,3,2,0,3,3,0};
//     int atom_type[] = {0,0,0,0};
//     int type_weight[] = {3};
//     int orb_type[] = {0,1,1};
//     int edge_shift_vec[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
//     double pos[] = {0.0,0.5,0.5,0.0,0.0,0.0,0.0,0.5,0.0,0.0,0.0,0.5};
//     DM dm;
//     MPI_Comm comm = MPI_COMM_WORLD;
//     rsatb::backend::get_dm_atomic(
//       n_node, n_edge, edge_index, atom_type, type_weight, edge_shift_vec, pos, &dm, comm
//     );

//     IS isSrc, isDst;

//     setup_is_orbital(dm, orb_type, 1, type_weight, &isSrc, &isDst);
// }

// // ------ FIX 2: provide a main that initialises PETSc & MPI ----------
// int main(int argc, char** argv)
// {
//     PetscInitialize(&argc, &argv, nullptr, nullptr);
//     c_dbcsr_init_lib(MPI_COMM_WORLD, nullptr);
//     ::testing::InitGoogleTest(&argc, argv);
//     int result = RUN_ALL_TESTS();
//     c_dbcsr_finalize_lib();
//     PetscFinalize();
//     return result;
// }
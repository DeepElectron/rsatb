// test_dm_atomic.cpp
#include <petscdmplex.h>
#include <mpi.h>
#include <cassert>
#include <iostream>
#include <vector>

// Your DMAtomic class
#include "petsc.hpp"

using namespace rsatb::backend;

// Helper to abort on error but print message first
#define CHECK(cond, msg)                                        \
  do {                                                          \
    if (!(cond)) {                                              \
      std::cerr << "TEST FAILURE: " << msg << std::endl;        \
      MPI_Abort(PETSC_COMM_WORLD, 1);                           \
    }                                                           \
  } while (0)


static void test_mappings(DMAtomic &dmA,
                          size_t n_atom_local,
                          size_t N_atom,
                          size_t N_edge,
                          const std::vector<int> &atom_type_global,
                          const std::vector<int> &type_norb,
                          MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);

  int vStart, vEnd;
  dmA.get_vertex_range_local(&vStart, &vEnd);

  // 1. Basic range sanity
  CHECK(vStart <= vEnd, "vertex range is inverted");

  // For debugging output
  std::cout << "[rank " << rank << "] vertex range: "
            << vStart << " .. " << vEnd << std::endl;

  // 2. Check local↔global vertex index consistency, atom index mappings,
  //    and ghost flags.
  int nGhost = 0;
  for (int v = vStart; v < vEnd; ++v) {
    int gvi   = dmA.mapvl2vg()[v];        // global vertex index (may be negative for ghosts)
    bool ghost = dmA.atom_is_ghost(v); // according to your class

    std::cout << "[rank " << rank << "] v=" << v
              << "  gvi=" << gvi
              << "  ghost=" << ghost << std::endl;

    if (gvi >= 0) {
      // Owned vertex: should not be marked ghost
      CHECK(!ghost, "owned vertex marked as ghost");

      // Inverse mapping must be consistent: GVI -> LVI
      int v_back = dmA.GVI2LVI(gvi);
      CHECK(v_back == v, "GVI2LVI(LVI2GVI(v)) != v");

      // Global atom index from both paths must match and be in range
      int gai1 = dmA.LVI2GAI(v);   // via local
      int gai2 = dmA.GVI2GAI(gvi); // via global vertex
      CHECK(gai1 == gai2, "LVI2GAI != GVI2GAI");
      CHECK(gai1 >= 0 && gai1 < (int)N_atom,
            "global atom index out of range for owned vertex");

      // Local atom index mapping for owned vertices must be in [0, n_atom_local)
      int lai1 = dmA.LVI2LAI(v);
      int lai2 = dmA.GVI2LAI(gvi);
      CHECK(lai1 == lai2, "LVI2LAI != GVI2LAI for owned vertex");
      CHECK(lai1 >= 0 && lai1 < (int)n_atom_local,
            "local atom index out of range for owned vertex");

      // Round trips GAI ↔ LVI and LAI ↔ LVI must be consistent
      CHECK(dmA.GAI2LVI(gai1) == v, "GAI2LVI(LVI2GAI(v)) != v");
      CHECK(dmA.LAI2LVI(lai1) == v, "LAI2LVI(LVI2LAI(v)) != v");

      // (Optional) you can also check type/dof here once _ps_ is debug-accessible

    } else {
      // Ghost vertex
      nGhost++;

      CHECK(ghost, "ghost vertex not marked as ghost by atom_is_ghost");

      // For ghosts we *can't* require LAI to be in [0, n_atom_local),
      // but we can at least check GAI is in [0, N_atom) if your mappings
      // are implemented that way.
      int gai = dmA.LVI2GAI(v); // may not be meaningful yet in your current code
      // If LVI2GAI is supposed to work for ghosts, uncomment:
      // CHECK(gai >= 0 && gai < (int)N_atom,
      //       "global atom index out of range for ghost vertex (if defined)");
    }
  }

  // 3. Simple sanity: we expect exactly one ghost on each rank for this graph.
  if (N_atom == 4 && N_edge == 3) {
    if (rank == 0 || rank == 1) {
      CHECK(nGhost == 1,
            "expected exactly one ghost vertex on rank "
            << rank << ", got " << nGhost);
    }
  }

  // You can add more checks here once you expose DOF info from the PetscSection.
}

int main(int argc, char **argv)
{
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  MPI_Comm comm = PETSC_COMM_WORLD;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (size != 2) {
    if (rank == 0) {
      std::cerr << "This test is written for 2 MPI ranks. "
                   "Run with: mpirun -n 2 ./test_dm_atomic\n";
    }
    PetscFinalize();
    return 0;
  }

  // ---- Define global problem ----
  const size_t N_atom = 4;
  const size_t N_edge = 4;

  // atom types and norb (global)
  std::vector<int> atom_type_global = {0, 1, 1, 0};
  std::vector<int> type_norb        = {2, 3}; // type 0 -> 2, type 1 -> 3

  // Local atom count and offsets
  size_t n_atom   = 2;
  size_t offset   = (rank == 0) ? 0 : 2; // [0,1] on rank 0, [2,3] on rank 1

  // Local atom_type slice
  std::vector<int> atom_type_local(n_atom);
  for (size_t i = 0; i < n_atom; ++i) {
    atom_type_local[i] = atom_type_global[offset + i];
  }

  // Local edge list (global vertex indices)
  size_t n_edge;
  std::vector<int> edge_index, atom_index; // (src,dst) pairs

  if (rank == 0) {
    // edges: 0-1 and 1-2
    n_edge = 2;
    edge_index = {0, 1,  1, 2};
    atom_index = {0, 1};
  } else {
    // edges: 2-3
    n_edge = 2;
    edge_index = {2, 3, 2, 1};
    atom_index = {2, 3};
  }

  // Dummy cell & pos (not used in your current DMAtomic constructor)
  double cell[9] = {1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0};

  // For simplicity we only pass local positions; your DMAtomic currently
  // just stores the pointer and doesn't use it, so this is fine for now.
  double pos_local[3 * 2] = {
      (double)(offset + 0), 0.0, 0.0,
      (double)(offset + 1), 0.0, 0.0};

  // No shift vectors for this simple test
  const int *edge_shift_vec = nullptr;

  try {
    // ---- Construct DMAtomic ----
    DMAtomic dmA(n_atom, N_atom, offset,
                 n_edge, N_edge,
                 atom_index.data(),
                 atom_type_local.data(),
                 edge_index.data(),
                 type_norb.data(),
                 edge_shift_vec,
                 cell,
                 pos_local,
                 comm);

    // ---- Run mapping tests ----
    test_mappings(dmA,
                  n_atom,
                  N_atom,
                  N_edge,
                  atom_type_global,
                  type_norb,
                  comm);

    if (rank == 0) {
      std::cout << "All DMAtomic mapping tests PASSED." << std::endl;
    }
  } catch (const std::exception &ex) {
    std::cerr << "[rank " << rank
              << "] DMAtomic threw exception: " << ex.what() << std::endl;
    MPI_Abort(comm, 1);
  }

  PetscFinalize();
  return 0;
}

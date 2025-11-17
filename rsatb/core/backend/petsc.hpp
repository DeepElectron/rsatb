#ifndef petsc
#define petsc

#include <stdexcept>
#include <petsc.h>
#include <mpi.h>
#include <string>

namespace rsatb::backend{

static inline PetscErrorCode
int2petscint(int n, const int *in,
             PetscInt **out,         /* ← result */
             PetscBool *needs_free)  /* ← says if caller must PetscFree(*out) */
{
  PetscFunctionBegin;
  if (sizeof(PetscInt) == sizeof(int)) {
    *out = (PetscInt*)in;            /* just alias */
    if (needs_free) *needs_free = PETSC_FALSE;
  } else {
    PetscCall(PetscMalloc1(n, out)); /* allocate now the pointer value *out is changed */
    for (PetscInt k=0; k<(PetscInt)n; ++k) (*out)[k] = (PetscInt)in[k];
    if (needs_free) *needs_free = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* same idea for double → PetscScalar ------------------------------- */
static inline PetscErrorCode
double2petscscalar(int n, const double *in,
                   PetscScalar **out,
                   PetscBool *needs_free)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscMalloc1(n,out));
  for (PetscInt k=0; k<(PetscInt)n; ++k) (*out)[k] = PetscCMPLX(in[k],0.0);
  if (needs_free) *needs_free = PETSC_TRUE;
#else
  *out = (PetscScalar*)in;           /* alias – types have same size */
  if (needs_free) *needs_free = PETSC_FALSE;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode
cdouble2petscscalar(int n, const std::complex<double> *in,
                   PetscScalar **out,
                   PetscBool *needs_free)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  *out = (PetscScalar*)in;           /* alias – types have same size */
  if (needs_free) *needs_free = PETSC_FALSE;
#else
  PetscCall(PetscMalloc1(n,out));
  for (PetscInt k=0; k<(PetscInt)n; ++k) (*out)[k] = in[k].real;
  if (needs_free) *needs_free = PETSC_TRUE;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode
double2petscreal(int n, const double *in,
                   PetscReal **out,
                   PetscBool *needs_free)
{
  PetscFunctionBegin;
  if (sizeof(PetscReal) == sizeof(double)) {
    *out = (PetscReal*)in;            /* just alias */
    if (needs_free) *needs_free = PETSC_FALSE;
  } else {
    PetscCall(PetscMalloc1(n, out)); /* allocate now the pointer value *out is changed */
    for (PetscInt k=0; k<(PetscInt)n; ++k) (*out)[k] = (PetscReal)in[k];
    if (needs_free) *needs_free = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct AtomData{
        PetscReal x, y, z;
        PetscInt index, type, nR, NR;
        PetscInt *R;
    };

struct BondData{
        PetscInt Rx, Ry, Rz; // TODO: optimizable, can stored all repetitive edge into one to avoid traffic in hr2hk.
    };

PetscErrorCode get_dm_atomic(
            const int n_node,
            const int n_edge,
            const int *edge_index,
            const int *atom_type, 
            const int *type_weight,
            const int *edge_shift_vec,
            const double *pos,
            DM *dm,
            MPI_Comm comm
            );

PetscErrorCode form_local_edge_list(
            DM dm,
            int *src,
            int *tgt,
            MPI_Comm comm
);

template<typename T>
void get_attr_from_dm(
            const DM dm,
            const std::string tp,
            const int idx,
            // output
            T *attr,
            int *nvar
);

PetscErrorCode ViewDMNetworkPartition(DM dm, bool detail);

void get_edge_index_from_dm(
    const DM dm,
    const int idx,
    // output
    int *i,
    int *j
);

void setup_KSP(
    DM dm, 
    Mat S,
    int nrhs,
    MPI_Comm comm,
    //output
    KSP *ksp
);

void setup_is_atomic(const DM dm, IS *isSrc, IS *isDst);

void setup_is_orbital(const DM dm, const int *orbtype, const int &n_atom_type, const int *type_weight, IS *isSrc, IS *isDst);

/* The following part will be a DMPLEX version of atomic structure DM*/

class DMAtomic
{
private:
    DM _dm_;
    PetscSection _ps_, _psg_, _psid_, _pspos_, _psesv_;
    PetscSF _sf_;
    int _rank_, _size_, _ntype_;
    int *_atom_type_, *_atom_index_, *_edge_shift_vec_;
    double *_pos_;

public:
    DMAtomic(
      const size_t n_atom, const size_t N_atom, const size_t atom_offset, const size_t n_edge, const size_t N_edge,
      const int *atom_type, const int *edge_index, const int *type_norb, const int *edge_shift_vec,
      const double *cell, const double *pos,
      MPI_Comm comm
    ) 
    // the atom index will be ( atom_offset, atom_offset + n_atom )
    /* We need to assert that:
      1. The input atoms are divided into continuous block in each process, so the atom index for each are: ( atom_offset, atom_offset + n_atom ), so
        - the N_atom = sum of all n_atom
        - the N_edge = sum of all n_edge
      2. The edge_index (src, dst), are seperated where src contains only atoms in the seperated section defined in 1
      3. atom_type are continuous, it must start from 0, 1, 2, 3, no gap is allowed.
      4. type_norb have length of max(atom_type)+1
      */ 
      _type_norb_(type_norb),
      _ntype_((*std::max_element(atom_type, atom_type+n_atom))+1)
    {
      // build DM
      MPI_Comm_rank(comm, &_rank_);
      MPI_Comm_size(comm, &_size_);

      DMPlexCreate(comm, &dm);
      _dm_ = NULL;

      PetscSection psid, pspos, psesv;
      PetscInt pStart, pEnd, vStart, vEnd, eStart, eEnd;
      int N=n_atom;

      // check the input
        // the input numbers of edge/atom
      MPI_Allreduce(MPI_IN_PLACE, &N, 1, MPI_INT, MPIU_SUM, comm);
      if (N != N_atom) {
        throw std::runtime_error("The input number of atoms might be overlapped or absent!");
      }
      N = n_edge;
      MPI_Allreduce(MPI_IN_PLACE, &N, 1, MPI_INT, MPIU_SUM, comm);
      if (N != N_edge) {
        throw std::runtime_error("The input number of edge might be overlapped or absent!");
      }
        // the input edge index
      
      for(int i=0; i<2*n_edge; i+=2){
        if (edge_index[i]<atom_offset || edge_index[i]>=atom_offset+n_atom) {
          throw std::runtime_error("The input edge_index contains atoms in other process!");
        }
      }

        // type info
      MPI_Allreduce(MPI_IN_PLACE, &_ntype_, 1, MPI_INT, MPIU_MAX, comm);

      if (_rank_ == 0) {
        if ( n_atom == N_atom ) {
          DMPlexBuildFromCellList(dm, N_edge, N_atom, 2, edge_index); // sequential build
        } else {
          if (_size_ == 1) {
            throw std::runtime_error("The input atomic graph is not complete in a single process construction.");
          }
          DMPlexBuildFromCellListParallel(dm, n_edge, n_atom, N_atom, 2, edge_index, NULL, NULL);
        }
      } else {
        if ( n_atom == N_atom ) {
          DMPlexBuildFromCellList(dm, 0, 0, 2, NULL);
        } else {
          DMPlexBuildFromCellListParallel(dm, n_edge, n_atom, N_atom, 2, edge_index, NULL, NULL);
        }
      }

      DMPlexGetChart(dm, &pStart, &pEnd);
      DMPlexGetHeightStratum(dm, 0, &eStart, &eEnd);   // edges
      DMPlexGetHeightStratum(dm, 1, &vStart, &vEnd);   // vertices

      PetscSectionCreate(psid);
      PetscSectionCreate(pspos);
      PetscSectionCreate(psesv);
      PetscSectionSetChart(psid, pStart, pEnd);
      PetscSectionSetChart(pspos, pStart, pEnd);
      PetscSectionSetChart(psesv, pStart, pEnd);

      for(PetscInt v=vStart; v<vEnd; ++v) {
        PetscSectionSetDof(psid, v, 1);
        PetscSectionSetDof(pspos, v, 3);
      }

      for(PetscInt e=eStart; e<eEnd; ++e) {
        PetscSectionSetDof(psesv, e, 3);
      }

      PetscSectionSetUp(psid); PetscSectionSetUp(pspos); PetscSectionSetUp(psesv);

      // distribute and handle the attribute of node and edges
      std::vector<int> atom_index(n_atom, 0);
      for (int i=atom_offset; i<atom_offset+n_atom; ++i) atom_index[i-atom_offset] = i;

      if (_size_ > 1) {
        DMPlexDistribute(dm, 0, &_sf_, &_dm_);

        DMPlexDistributeData(_dm_, _sf_, psid, MPI_INT, atom_index.data(), _psid_, &_atom_index_);
        DMPlexDistributeData(_dm_, _sf_, psid, MPI_INT, atom_type, _psid_, &_atom_type_);
        DMPlexDistributeData(_dm_, _sf_, psesv, MPI_INT, edge_shift_vec, _psesv_, &_edge_shift_vec_);
        DMPlexDistributeData(_dm_, _sf_, pspos, MPI_DOUBLE, pos, _pspos_, &_pos_);

        PetscSectionDestroy(&ps); PetscSectionDestroy(&psid); PetscSectionDestroy(&psesv); PetscSectionDestroy(&pspos);
        DMDestroy(&dm);
      } else {
        _dm_ = dm;
        _psid_ = psid;
        _psesv_ = psesv;
        _pspos_ = pspos;
        DMGetPointSF(_dm_, &_sf_);
        
        _atom_index_ = new int[N_atom];
        for (int i=0; i<N_atom; ++i) _atom_index_[i] = atom_index[i];
        _atom_type_ = atom_type;
        _edge_shift_vec_ = edge_shift_vec;
        _pos_ = pos;
      }

      // construct the PETScSection for new distributed DM
      DMPlexGetChart(_dm_, &pStart, &pEnd);
      DMPlexGetHeightStratum(_dm_, 0, &eStart, &eEnd);   // edges
      DMPlexGetHeightStratum(_dm_, 1, &vStart, &vEnd);   // vertices

      PetscSectionCreate(_ps_);
      PetscSectionSetChart(_ps_, pStart, pEnd);
      for(PetscInt v=vStart; v<vEnd; ++v) {
        PetscSectionSetDof(_ps_, v, _type_norb_[_atom_type_[v-vStart]]);
      }
      PetscSectionSetUp(_ps_);
      DMSetLocalSection(_ps_);
      DMGetGlobalSection(_psg_);
    };

    ~DMAtomic() {};

    /*
     Problems to be consider seriously before inplementing the functions
     1. What will the distributed DM looks like? Is every vertice in that local dm owned? where is the ghost? how to tell the difference?
     2. Is the ghost and owned vertices mixed? or continuously layout seperately?
     3. Does process k owned all the i vertices in its edge i-j? Is edge distributed with no repetition?
     4. How to build map between three index system? The global index, local index (in PETSc), and the original atom_index
     5. What are the function needed?
    */

    void get_atom_range(int *start, int *end) {};

    bool atom_is_ghost(int idx){
      PetscInt ghost;
      PetscInt *leaves=nullptr, *remote=nullptr;
      PetscSFGetGraph(_sf_, NULL, &Nl, &leaves, &remote);
      PetscFindInt(idx, Nl, leaves, &ghost);
      f (ghost >= 0) {
        // Vertex is not owned
        return true;
      } else {
        // Vertex is owned
        return false;
      }
    }

    void get_atom_offset_local(int idx, size_t *offset) {
      PetscSectionGetOffset(_ps_, idx, reinterpret_cast<PetscInt*>(offset));
    };

    void get_atom_offset_global(int idx, size_t *offset) {};

    void get_atom_pos(int idx, double *x, double *y, double *z) {};

    void get_atom_type(int idx, int *atype) {};

    void get_atom_norb(int idx, int *norb) {};

    void atom_is_ghost(int idx, bool *ghost) {};

    void get_bond_range(int *start, int *end) {};

    void get_shift_vec(int idx, int *Rx, int *Ry, int *Rz) {};

    void get_connected_atoms(int idx, int src, int dst) {}; // [i(src), j(dst)]

};

PetscErrorCode get_dm_atomic(
            const int n_node,
            const int n_edge,
            const int *edge_index,
            const int *atom_type, 
            const int *type_weight,
            const int *edge_shift_vec,
            const double *pos,
            DM *dm,
            MPI_Comm comm,

          );

}

#endif
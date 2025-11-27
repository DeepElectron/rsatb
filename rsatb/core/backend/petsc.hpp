#ifndef petsc
#define petsc

#include <stdexcept>
#include <petsc.h>
#include <mpi.h>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include "petsc_hack.h"

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
/* The following part will be a DMPLEX version of atomic structure DM*/

class DMAtomic
{
private:
    DM _dm_;
    PetscSection _ps_, _psg_;
    PetscSF _sf_;
    IS pNum;
    int _rank_, _size_, _n_type_;
    int *_atom_index_, *_atom_type_;
    const int *_edge_index_, *_edge_shift_vec_, *_type_norb_;
    double *_pos_, *_x_, *_y_, *_z_;
    const double *_cell_;
    size_t _atom_offset_, _n_edge_, _N_edge_, _n_atom_, _N_atom_;
    PetscInt pStart, pEnd, vStart, vEnd, eStart, eEnd;
    std::unordered_map<int, int> vl2vg, vg2vl;

public:
    DMAtomic()
    {};

    DMAtomic(const DMAtomic&) = delete;
    DMAtomic& operator=(const DMAtomic&) = delete;

    DMAtomic(
      const size_t n_atom, const size_t N_atom, const size_t atom_offset, const size_t n_edge, const size_t N_edge,
      const int *atom_index, const int *atom_type, const int *edge_index, const int *type_norb, const int *edge_shift_vec,
      const double *cell, const double *pos,
      MPI_Comm comm
    ):
    // the atom index will be ( atom_offset, atom_offset + n_atom )
    /* We need to assert that:
      1. The input atoms are divided into continuous block in each process, so the atom index for each are: ( atom_offset, atom_offset + n_atom ), so
        - the N_atom = sum of all n_atom
        - the N_edge = sum of all n_edge
      2. The edge_index (src, dst), are seperated where src contains only atoms in the seperated section defined in 1
      3. atom_type are continuous, it must start from 0, 1, 2, 3, no gap is allowed.
      4. type_norb have length of max(atom_type)+1
      5. the atom_offset should equals to the prefix summation of n_atom in other process, e.g. for process 2, the atom_offset = n_atom(0) + n_atom(1)
      */
      _type_norb_(type_norb), _edge_index_(edge_index), _edge_shift_vec_(edge_shift_vec),
      _n_type_((*std::max_element(atom_type, atom_type+n_atom))+1), _cell_(cell),
      _atom_offset_(atom_offset), _n_edge_(n_edge), _N_edge_(N_edge), _n_atom_(n_atom), _N_atom_(N_atom)
    {
      // build DM
      MPI_Comm_rank(comm, &_rank_);
      MPI_Comm_size(comm, &_size_);

      DMPlexCreate(comm, &_dm_);

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
      MPI_Allreduce(MPI_IN_PLACE, &_n_type_, 1, MPI_INT, MPIU_MAX, comm);

       // offset
      int atom_offset_check = 0;
      MPI_Exscan(&n_atom, &atom_offset_check, 1, MPI_INT, MPI_SUM, comm);
      if ( atom_offset != atom_offset_check) {
        throw std::runtime_error("The input atom offset is not correct!");
      }

      HACKDMPlexBuildFromCellListParallel(_dm_, n_edge, n_atom, N_atom, 2, edge_index, NULL, NULL);
      

      DMGetPointSF(_dm_, &_sf_);
      DMPlexCreatePointNumbering(_dm_, &pNum); 
      /*
        according to https://petsc.org/release/manualpages/DMPlex/DMPlexBuildFromCellListParallel/#dmplexbuildfromcelllistparallel
        The numbering will first be edges, then the vertices globally.
        So we should expect: edge numbering - [0, N_edge)
                             vertices numbering - [N_edge, N_edge+N_atom)
        Therefore, we now get the correspondance of global vertice index to atom index, with an offset by N_edge.
        What left to build is the local petsc index to global vertice index
      */

      DMPlexGetChart(_dm_, &pStart, &pEnd);
      DMPlexGetHeightStratum(_dm_, 0, &eStart, &eEnd);   // edges
      DMPlexGetHeightStratum(_dm_, 1, &vStart, &vEnd);   // vertices

      PetscSectionCreate(comm, &_ps_);
      PetscSectionSetChart(_ps_, pStart, pEnd);

      PetscInt ghost, idxglobal, Nl, Nr;
      PetscInt *atype_leaf=nullptr, *atype_root=nullptr;
      const PetscInt *ilocal=nullptr, *pGlobal=nullptr;
      const PetscSFNode *remote=nullptr;

      PetscSFGetGraph(_sf_, &Nr, &Nl, &ilocal, &remote); /* Nr is non-repetitive point that is actually owned by this rank
                                                           Nl includes both point that owned and the ghost point, all of them contains all kind of point in DMPlex*/ 
      // std::cout << _rank_ << " " << Nl << " " << Nr << " " << n_atom << " " << n_edge << " " << pEnd << " " << pStart << std::endl;
      
      if ( Nr != pEnd - pStart ) { // now, Nr might contains ghost as well
        throw std::runtime_error("The PetscSF num of root is not correct! or I understand it wrontgly.");
      }

      // std::cout << "Here? " << Nr << " " << Nl << " " << pEnd - pStart - n_atom - n_edge << " "<< Nr - n_edge << " " << n_atom << " " << N_atom << " " << vEnd << " " << vStart << std::endl;
      // for (int i=0; i<Nl; ++i) {
      //   std::cout << ilocal[i] << " \n" << std::endl;
      // }

      if ( Nl != pEnd - pStart - n_atom - n_edge ){ // Nl might should be the number of ghost exist
        throw std::runtime_error("The PetscSF num of leaves is not correct! or I understand it wrontgly.");
      }

      

      int leaf_size=Nl;

      if (ilocal != NULL) {
        leaf_size = (*std::max_element(reinterpret_cast<const int*>(ilocal), reinterpret_cast<const int*>(ilocal)+Nl))+1;
      }

      

      ISGetIndices(pNum, &pGlobal);


      // Bcast to get the ghost vertices's atom attributes: atom_type, atom_index, pos
      // std::cout << vEnd - vStart << " Mem length!" << std::endl;
      _x_ = new double[vEnd-vStart];
      _y_ = new double[vEnd-vStart];
      _z_ = new double[vEnd-vStart];
      _atom_type_ = new int[vEnd-vStart];
      _atom_index_ = new int[vEnd-vStart];

      int* w_ptr[3] = {_atom_type_, _atom_index_, nullptr};
      const int* d_ptr[3] = {atom_type, atom_index, nullptr};

      PetscMalloc1(Nr, &atype_root);
      PetscMalloc1(leaf_size, &atype_leaf);
      for(int iptr=0; iptr<2; ++iptr) {
        for(PetscInt v=vStart; v<vEnd; ++v) {

          PetscFindInt(v-pStart, Nl, ilocal, &ghost); // ilocal shape pEnd-pStart
          idxglobal = pGlobal[v];
          if (idxglobal > 0){ // not ghost
            atype_root[v] = d_ptr[iptr][idxglobal-N_edge-atom_offset];
          }
        }

        PetscSFBcastBegin(_sf_, MPIU_INT, atype_root, atype_leaf, MPI_REPLACE);
        PetscSFBcastEnd(_sf_, MPIU_INT, atype_root, atype_leaf, MPI_REPLACE);

        if (leaf_size > 0) {
          for (PetscInt i = 0; i < Nl; ++i) {
            // std::cout << ilocal[i] << " ilocal i first" << std::endl;
            if (ilocal == NULL){
              atype_root[i] = atype_leaf[i];
            } else {
              atype_root[ilocal[i]] = atype_leaf[ilocal[i]];
            }
          }
        }

        for(PetscInt v=vStart; v<vEnd; ++v) {
          w_ptr[iptr][v-vStart] = atype_root[v-pStart];
        }
      }

      // std::cout << "Before first free" << std::endl;
      if (Nr > 0){
        PetscFree(atype_root);
      }
      if (leaf_size > 0){
        PetscFree(atype_leaf);
      }
      // std::cout << "After first free" << std::endl;

      PetscReal *atype_leaf_=nullptr, *atype_root_=nullptr;
      double* w_ptr_[3] = {_x_, _y_, _z_};

      PetscMalloc1(Nr, &atype_root_);
      PetscMalloc1(leaf_size, &atype_leaf_);

      for(int iptr=0; iptr<3; ++iptr) {
        for(PetscInt v=vStart; v<vEnd; ++v) {

          PetscFindInt(v-pStart, Nl, ilocal, &ghost); // ilocal shape pEnd-pStart
          idxglobal = pGlobal[v];
          if (idxglobal > 0){ // not ghost
            atype_root_[v] = pos[3*(idxglobal-N_edge-atom_offset)+iptr];
          }
        }

        PetscSFBcastBegin(_sf_, MPI_DOUBLE, atype_root_, atype_leaf_, MPI_REPLACE);
        PetscSFBcastEnd(_sf_, MPI_DOUBLE, atype_root_, atype_leaf_, MPI_REPLACE);

        if (leaf_size > 0) {
          for (PetscInt i = 0; i < Nl; ++i) {
            // std::cout << ilocal[i] << " ilocal i second" << std::endl;
            if (ilocal == NULL){
              atype_root_[i] = atype_leaf_[i];
            } else {
              atype_root_[ilocal[i]] = atype_leaf_[ilocal[i]];
            }
          }
        }

        for(PetscInt v=vStart; v<vEnd; ++v) {
          w_ptr_[iptr][v-vStart] = atype_root_[v-pStart];
        }
      }
      
      // std::cout << "Before SEcond free" << std::endl;
      if (Nr > 0){
        PetscFree(atype_root_);
      }
      if (leaf_size > 0){
        PetscFree(atype_leaf_);
      }
      // std::cout << "After SEcond free" << std::endl;

      for(PetscInt v=vStart; v<vEnd; ++v) {
        idxglobal = pGlobal[v]; // local atom index = global vertice index - N_edge - atom_offset
          // according to https://petsc.org/release/manualpages/DMPlex/DMPlexCreatePointNumbering/#dmplexcreatepointnumbering
          // negative idxglobal will be ghost point with index -(idx+1)
        // std::cout << "Here? " << v << _rank_ << " " << atype_root[v-pStart] << std::endl;
        PetscSectionSetDof(_ps_, v, _type_norb_[_atom_type_[v-vStart]]);

        vl2vg[v] = idxglobal;
        if (idxglobal < 0) {
          idxglobal = -idxglobal - 1;
          vg2vl[idxglobal] = -v-1;
        } else {
          vg2vl[idxglobal] = v;
        }
        
      }
      PetscSectionSetUp(_ps_);
      DMSetLocalSection(_dm_, _ps_);

      DMGetGlobalSection(_dm_, &_psg_);
    };

    ~DMAtomic() {
      delete[] _x_;
      delete[] _y_;
      delete[] _z_;
      delete[] _atom_index_;
      delete[] _atom_type_;
    };

    /*
     Problems to be consider seriously before inplementing the functions
     1. What will the distributed DM looks like? Is every vertice in that local dm owned? where is the ghost? how to tell the difference?
     2. Is the ghost and owned vertices mixed? or continuously layout seperately?
     3. Does process k owned all the i vertices in its edge i-j? Is edge distributed with no repetition?
     4. How to build map between three index system? The global index, local index (in PETSc, with ghost), and the local and global atom_index (without ghost)
      - (i) filling HK: local indices of petsc -> local offset and norb = local index of the atom
      - (ii) petsc to dbcsr: local indices of petsc -> global indices of the atom
      - (iii) dbcsr to petsc: global indices of the atom -> local indices of the petsc
     5. What are the function needed?
    */
    /* vertex, edge are petsc index, 
      atom, bond are global atomic index
        atom_id, the original index of the atom that passed by atom_index, that can be different from the one of GAI.
        This is the id used in the h5 file to label the blocks.
      LAI: local atom index
      GAI: global atom index
      LVI: local petsc vertex index
      GVI: global petsc vertex index
    */

    void get_vertex_range_local(int *start, int *end) const {
      *start = vStart; *end = vEnd;
    };

    void get_edge_range_local(int *start, int *end) const {
      *start = eStart; *end = eEnd;
    };

    void get_connected_atoms(int idx, int *i, int *j) const {
      PetscInt coneSize;
      const PetscInt *cone = NULL;
      DMPlexGetConeSize(_dm_, idx, &coneSize);
      DMPlexGetCone(_dm_, idx, &cone);
      if (coneSize != 2) {
        throw std::runtime_error("The cone size for edge is not correct!");
      }
      *i = cone[0]; *j = cone[1];
    }
    // xxx2xxx function will always output the correct - 0 & positive index
    // the ghost will hand over to the atom_is_ghost function
    int LVI2GVI(int idx) {
      if (vl2vg[idx] < 0){
        return -vl2vg[idx] - 1;
      }
      return vl2vg[idx];
    }

    int GVI2LVI(int idx) {
      if (vl2vg[idx] < 0){
        return -vg2vl[idx] - 1;
      }
      return vg2vl[idx];
    }

    int GVI2GAI(int idx) {
      if (idx < 0){
        return -idx - 1 - _N_edge_;
      }
      return idx - _N_edge_;
    }

    int GVI2LAI(int idx) {
      if (idx < 0){
        return -idx - 1 - _N_edge_ - _atom_offset_;
      }
      return idx - _N_edge_ - _atom_offset_;
    }

    int LVI2GAI(int idx) {
      int index = vl2vg[idx];
      if ( index < 0 ) index = -index - 1;
      return index - _N_edge_;
    }

    int LVI2LAI(int idx) {
      int index = vl2vg[idx];
      if ( index < 0 ) index = -index - 1;
      return index - _N_edge_ - _atom_offset_;
    }

    int GAI2LVI(int idx) {
      return GVI2LVI(idx + _N_edge_);
    }

    int LAI2LVI(int idx) {
      return GVI2LVI(idx + _N_edge_ + _atom_offset_);
    }

    bool atom_is_ghost(int idx) { // idx is local vertices indices
      if (vl2vg.find(idx) == vl2vg.end()){
        throw std::runtime_error("The input idx is out of range of local atom index section!");
      } else if ( vl2vg[idx] < 0 )
      {
        return true;
      }

      return false;
    }

    bool GAI_is_owned(int idx) {
      // std::cout << vg2vl[idx+_N_edge_] << std::endl;
      if (vg2vl.find(idx+_N_edge_) == vg2vl.end()){
        return false;
      } else if (vg2vl[idx+_N_edge_] < 0) {
        return false;
      }

      return true;
    }

    void get_offset_local(int idx, int *offset) { // idx is local petsc vertice index
      PetscSectionGetOffset(_ps_, idx, reinterpret_cast<PetscInt*>(offset));
    };

    void get_atom_type(int idx, int *tp) {
      *tp = _atom_type_[idx-_n_edge_];
    }

    void get_atom_norb(int idx, int *norb) {
      *norb = _type_norb_[_atom_type_[idx-_n_edge_]];
    }

    void get_atom_id(int idx, int *aid) {
      *aid = _atom_index_[idx-_n_edge_];
    }

    void get_pos(int idx, double *x, double *y, double *z) {
      *x = _x_[idx-_n_edge_];
      *y = _y_[idx-_n_edge_];
      *z = _z_[idx-_n_edge_];
    }

    void get_edge_shift_vec(int idx, int *Rx, int *Ry, int *Rz) {
      // we assume local edge index will start from 0 to n_edge
      *Rx = _edge_shift_vec_[3*idx];
      *Ry = _edge_shift_vec_[3*idx+1];
      *Rz = _edge_shift_vec_[3*idx+2];
    }

    DM dm() { return _dm_; }

    size_t N_edge() { return _N_edge_; }
    size_t N_atom() { return _N_atom_; }
    size_t n_atom() { return _n_atom_; }
    size_t n_edge() { return _n_edge_; }
    size_t n_type() { return _n_type_; }
    int rank() { return _rank_; }
    const int* type_norb() { return _type_norb_; }
    const double* cell() { return _cell_; }
    double* pos() { return _pos_; }
    int* atom_type() { return _atom_type_; }
    const int* atom_norb() { return _type_norb_; }

    std::unordered_map<int, int> mapvl2vg() { return vl2vg; }
 
    // void get_edge_index(int idx, int *idx1, int *idx2) { // idx is the local index of the edges
      
    // }

    // void get_offset_global(int idx, size_t *offset) {};

    // void get_atom_pos(int idx, double *x, double *y, double *z) {};

    // void get_atom_type(int idx, int *atype) {};

    // void get_atom_norb(int idx, int *norb) {};

    // void atom_is_ghost(int idx, bool *ghost) {};

    // void get_bond_range(int *start, int *end) {};

    // void get_shift_vec(int idx, int *Rx, int *Ry, int *Rz) {};

    // void get_connected_atoms(int idx, int src, int dst) {}; // [i(src), j(dst)]

};

void setup_is_atomic(const DM dm, IS *isSrc, IS *isDst);

void setup_is_atomic(DMAtomic &dm, IS *isSrc, IS *isDst);

void setup_is_orbital(const DM dm, const int *orbtype, const int &n_atom_type, const int *type_weight, IS *isSrc, IS *isDst);

void setup_is_orbital(DMAtomic &dm, const int *orbtype, const int &n_atom_type, const int *type_weight, IS *isSrc, IS *isDst);



}

#endif
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
        PetscInt Rx, Ry, Rz;
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

}

#endif
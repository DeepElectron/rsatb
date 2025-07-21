#ifndef hr2hk
#define hr2hk

#include <complex>
#include "backend/petsc.hpp"
#include "io/hdf5.hpp"

// hr2hk petsc
namespace rsatb{


PetscErrorCode SetBlockR2K(Mat mat, DM dm, int i, int j, int inorb, int jnorb, int Rx, int Ry, int Rz, double Kx, double Ky, double Kz, double *v);

void setMatrixR2K(Mat mat, DM dm, rsatb::io::HDF5Handler *handler, double Kx, double Ky, double Kz);

void cal_phase(int Rx, int Ry, int Rz, double Kx, double Ky, double Kz, std::complex<double> *phase);

}

#endif
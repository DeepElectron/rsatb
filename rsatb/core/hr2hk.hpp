#ifndef hr2hk
#define hr2hk

#include <complex>
#include "backend/petsc.hpp"
#include "io/hdf5.hpp"
#include "extern/nao-abacus/source_basis/module_nao/two_center_bundle.h"

// hr2hk petsc
namespace rsatb{

PetscErrorCode SetBlockR2K(Mat mat, DM dm, int i, int j, int inorb, int jnorb, int Rx, int Ry, int Rz, double Kx, double Ky, double Kz, double *v);

void setMatrixR2K(Mat mat, DM dm, rsatb::io::HDF5Handler *handler, double Kx, double Ky, double Kz);

// used only for computing overlap and position operator currently.
void setMatrixR2K(Mat S, DM dm, const double* cell, int ntype, const std::string* file_orb0, const std::string& orbital_dir, double Kx, double Ky, double Kz, Mat Rx=nullptr, Mat Ry=nullptr, Mat Rz=nullptr);

void cal_phase(int Rx, int Ry, int Rz, double Kx, double Ky, double Kz, std::complex<double> *phase);

}

#endif
#ifndef hr2hk
#define hr2hk

#include <complex>
#include "backend/petsc.hpp"
#include "io/hdf5.hpp"
#include "extern/nao-abacus/source_basis/module_nao/two_center_bundle.h"
#include "extern/nao-abacus/source_estate/op2c.hpp"

// hr2hk petsc
namespace rsatb{

// PetscErrorCode SetBlockR2K(Mat mat, DM dm, int i, int j, int inorb, int jnorb, int Rx, int Ry, int Rz, double Kx, double Ky, double Kz, double *v);

PetscErrorCode SetLBlockR2K(Mat mat, rsatb::backend::DMAtomic &dm_atomic, int i, int j, int inorb, int jnorb, int Rx, int Ry, int Rz, double Kx, double Ky, double Kz, double *v, std::complex<double> factor=1);
PetscErrorCode SetGBlockR2K(Mat mat, rsatb::backend::DMAtomic &dm_atomic, int i, int j, int inorb, int jnorb, int Rx, int Ry, int Rz, double Kx, double Ky, double Kz, double *v, std::complex<double> factor=1);

// void setMatrixR2K(Mat mat, DM dm, rsatb::io::HDF5Handler *handler, double Kx, double Ky, double Kz);

void setMatrixR2K(Mat mat, rsatb::backend::DMAtomic &dm_atomic, rsatb::io::HDF5Handler *handler, double Kx, double Ky, double Kz);

// used only for computing overlap and position operator currently.
// void setMatrixR2K(Mat S, DM dm, const double* cell, int ntype, const std::string* file_orb0, const std::string& orbital_dir, double Kx, double Ky, double Kz, Mat Rx=nullptr, Mat Ry=nullptr, Mat Rz=nullptr);

void setMatrixR2K(Mat S, rsatb::backend::DMAtomic &dm_atomic, const double* cell, int ntype, const std::string* file_orb0, const std::string& orbital_dir, double Kx, double Ky, double Kz, Mat Rx=nullptr, Mat Ry=nullptr, Mat Rz=nullptr);

// used for computing overlap and velocity operator.
void setMatrixR2K(Mat S, rsatb::backend::DMAtomic &dm_atomic, const double* cell, int ntype, 
                const std::string* orb_names, const std::string& orb_dir, const std::string* psd_names, const std::string& psd_dir, 
                double Kx, double Ky, double Kz, Mat Vx=nullptr, Mat Vy=nullptr, Mat Vz=nullptr);

void cal_phase(int Rx, int Ry, int Rz, double Kx, double Ky, double Kz, std::complex<double> *phase);

}

#endif
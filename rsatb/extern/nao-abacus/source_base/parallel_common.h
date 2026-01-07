#ifndef W_ABACUS_DEVELOP_ABACUS_DEVELOP_SOURCE_MODULE_BASE_PARALLEL_COMMON_H
#define W_ABACUS_DEVELOP_ABACUS_DEVELOP_SOURCE_MODULE_BASE_PARALLEL_COMMON_H

#ifdef __MPI
#include <mpi.h>
#else
typedef int MPI_Comm;
#define MPI_COMM_WORLD 0
#endif
#include <complex>
#include <string>

namespace Parallel_Common
{
//(1) bcast array
void bcast_complex_double(std::complex<double>* object, const int n, MPI_Comm comm);
void bcast_string(std::string* object, const int n, MPI_Comm comm);
void bcast_double(double* object, const int n, MPI_Comm comm);
void bcast_int(int* object, const int n, MPI_Comm comm);
void bcast_char(char* object, const int n, MPI_Comm comm);

//(2) bcast single
void bcast_complex_double(std::complex<double>& object, MPI_Comm comm);
void bcast_string(std::string& object, MPI_Comm comm);
void bcast_double(double& object, MPI_Comm comm);
void bcast_int(int& object, MPI_Comm comm);
void bcast_bool(bool& object, MPI_Comm comm);

} // namespace Parallel_Common

#endif

#ifndef ATOM_PSEUDO_H
#define ATOM_PSEUDO_H

// #include "source_base/global_variable.h"
#include "source_base/vector3.h"
// #include "source_io/output.h"
#include "source_base/complexarray.h"
#include "source_base/complexmatrix.h"
#include "source_basis/module_nao/beta_radials.h"
#ifdef __MPI
#include <mpi.h>
#else
typedef int MPI_Comm;
#ifndef MPI_COMM_WORLD
#define MPI_COMM_WORLD 0
#endif
#endif
#include "pseudo.h"
#include <vector>

class Atom_pseudo : public pseudo
{
public:

	Atom_pseudo();
	~Atom_pseudo();

	// mohan add 2021-05-07
	ModuleBase::ComplexArray d_so; //(:,:,:), spin-orbit case
	ModuleBase::matrix d_real; //(:,:), non-spin-orbit case
	int nproj;
	int nproj_soc; // dimension of D_ij^so
	int itype = 0;
	std::vector<int> non_zero_count_soc = {0, 0, 0, 0};
	std::vector<std::vector<int>> index1_soc = {{}, {}, {}, {}};
	std::vector<std::vector<int>> index2_soc = {{}, {}, {}, {}};

    BetaRadials beta_radials;
    bool coulomb_potential = false;

    void setup_nonlocal(const ModuleBase::Logger& logger, const bool lspinorb=false, const int& nspin=1, MPI_Comm comm=MPI_COMM_WORLD);

	void set_d_so( // mohan add 2021-05-07
		ModuleBase::ComplexMatrix &d_so_in,
		const int &nproj_in,
		const int &nproj_in_so,
		const bool has_so,
		const bool lspinorb=false,
		const int& nspin=1
		);


	inline void get_d(const int& is, const int& p1, const int& p2, const std::complex<double>*& tmp_d)
	{
		tmp_d = &this->d_so(is, p1, p2);
		return;
	}
	inline void get_d(const int& is, const int& p1, const int& p2, const double*& tmp_d)
	{
		tmp_d = &this->d_real(p1, p2);
		return;
	}
	

#ifdef __MPI
	void bcast_atom_pseudo(MPI_Comm comm=MPI_COMM_WORLD); // for upf201
#endif

};

#endif
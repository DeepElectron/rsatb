#ifndef READ_PSEUDO_H
#define READ_PSEUDO_H

// #include "source_cell/unitcell.h"
// #include "source_cell/cal_atoms_info.h"

// #include "source_cell/atom_spec.h"
#include "source_base/log.h"
#include "source_cell/atom_pseudo.h"
#include <vector>

#ifdef __MPI
#include <mpi.h>
#else
typedef int MPI_Comm;
#ifndef MPI_COMM_WORLD
#define MPI_COMM_WORLD 0
#endif
#endif

// so the pseudo reading logic is, elecstate's function initialize a reader calss Pseudopot_upf, which calls the read function to give the value of 
// Atom_pseudo as an attribute to each atom. This partially setup the PP. It lacks something, especially, the projector, i.e. numerical beta is not initialized
// The projector are initilized using setupNonlocal in source_cell, 
void read_pseudo(const std::string& pp_dir, const std::vector<std::string> pp_name, const std::string& pp_type, 
const double rcut, const bool lspinorb, const double& soc_lambda,
const ModuleBase::Logger& logger, std::vector<Atom_pseudo>& pseudos, MPI_Comm comm);


void read_atom_pseudopots(const std::string& pp_dir, const std::string& pp_name, const std::string& pp_type, const double rcut, const bool lspinorb, const double& soc_lambda, const ModuleBase::Logger& logger, Atom_pseudo& atom_pseudo, MPI_Comm comm);

// read in pseudopotential from files for each type of atom
// void read_cell_pseudopots(const std::string& fn, std::ofstream& log, UnitCell& ucell);

// void print_unitcell_pseudo(const std::string& fn, UnitCell& ucell);

//===========================================
// calculate the total number of local basis
// Target : nwfc, lmax,
// 			atoms[].stapos_wf
// 			PARAM.inp.nbands
//===========================================
// void cal_nwfc(std::ofstream& log, UnitCell& ucell,Atom* atoms);

//======================
// Target : meshx
// Demand : atoms[].msh
//======================
void cal_meshx(int& meshx, const std::vector<Atom_pseudo>& pseudos);

//=========================
// Target : natomwfc
// Demand : atoms[].nchi
// 			atoms[].lchi
// 			atoms[].oc
// 			atoms[].na
//=========================
// void cal_natomwfc(std::ofstream& log,int& natomwfc,const int ntype,const Atom* atoms);

#endif
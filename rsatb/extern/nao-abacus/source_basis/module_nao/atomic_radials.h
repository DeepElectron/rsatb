#ifndef ATOMIC_RADIALS_H_
#define ATOMIC_RADIALS_H_

#include "source_basis/module_nao/radial_set.h"
#include "source_base/parallel_common.h"
#include "source_base/log.h"

//! The radial part of numerical atomic orbitals of a single element
/*!
 *  This class represents the radial part of all numerical atomic orbitals
 *  of a single element as read from an orbital file.
 *
 *  @see RadialSet
 *
 *  Usage:
 *
 *      int element_index = 1;
 *      std::ofstream ofs_log("/path/to/log/file");
 *      std::string orb_file = "/path/to/orbital/file";
 *
 *      AtomicRadials O_radials;
 *      O_radials.build(orb_file, element_index, ofs_log, GlobalV::MY_RANK);
 *
 *                                                                          */
class AtomicRadials : public RadialSet
{
  public:
    AtomicRadials() {}
    AtomicRadials(const AtomicRadials& other) : RadialSet(other), orb_ecut_(other.orb_ecut_) {}

    AtomicRadials& operator=(const AtomicRadials& rhs);
    AtomicRadials* clone() const { return new AtomicRadials(*this); } // covariant return type
    AtomicRadials* create_empty() const { return new AtomicRadials(); }

    ~AtomicRadials() {} // ~RadialSet() is called automatically

    //! Build the class from an orbital file
    void build(const std::string& file,          //!< orbital file name
               const int itype = 0,              //!< element index in calculation
               const int p = 0,
               const int pm = 0,                 //!< add an extra pm integer value to the readed l
               const ModuleBase::Logger* ptr_logger = nullptr, //!< output file stream for logging
               MPI_Comm comm = MPI_COMM_WORLD   //!< MPI comm
    );

    void build(const RadialSet* const other, const int itype, const int p = 0, const int pm = 0, const double rcut = -1.0) override;

    //! Get the energy cutoff as given by the orbital file
    double orb_ecut() const { return orb_ecut_; }

  private:
    double orb_ecut_; //!< energy cutoff as given by the orbital file

    //! Read the orbital file in the ABACUS format
    void read_abacus_orb(std::ifstream& ifs,               //!< input file stream from orbital file
                         const int p = 0, 
                         const int pm = 0,        // !< add pm to the readed l
                         const ModuleBase::Logger* ptr_logger = nullptr, //!< output file stream for logging
                         MPI_Comm comm = MPI_COMM_WORLD   //!< MPI comm
    );

};

#endif

#ifndef BETA_RADIALS_H_
#define BETA_RADIALS_H_

#include "source_basis/module_nao/radial_set.h"
#include "source_base/log.h"

//! The radial part of beta functions of a single element
/*!
 *  This class represents the radial part of all Kleinman-Bylander beta
 *  functions of a single element as read from a pseudopotential file.
 *
 *  @see RadialSet
 *
 *  Usage:
 *
 *      int element_index = 1;
 *      std::ofstream ofs_log("/path/to/log/file");
 *      std::string upf_file = "/path/to/pseudopotential/file";
 *
 *      BetaRadials O_beta;
 *      O_beta.build(orb_file, element_index, ofs_log, GlobalV::MY_RANK);
 *
 *                                                                          */
class BetaRadials : public RadialSet
{
  public:
    BetaRadials() {}
    BetaRadials(const BetaRadials& other) : RadialSet(other) {} //!< deep copy

    using RadialSet::operator=;
    BetaRadials* clone() const { return new BetaRadials(*this); } // covariant return type
    BetaRadials* create_empty() const { return new BetaRadials(); }

    ~BetaRadials() {}

    /// Build the class from a Numerical_Nonlocal object
    // void build(const Numerical_Nonlocal& nl,
    //            const int itype = 0,
    //            const ModuleBase::Logger* const ptr_logger = nullptr);

    // zyzh: added for avoid the inclusion of previous 2c implementation.
    void build(const NumericalRadial* nl, const int nchi, const int itype, const ModuleBase::Logger* const ptr_logger, MPI_Comm comm = MPI_COMM_WORLD);

    void build(const RadialSet* const other, const int itype, const int p = 0, const int pm = 0, const double rcut = -1.0) override;

    
    void build(const std::string& file,          //!< pseudopotential file name
               const bool lspinorb,              //!< whether to use spin-orbit interaction (for UPF v2.0.1)
               const int itype = 0,              //!< element index in calculation
               const int p = 0,
               const int pm = 0,
               const ModuleBase::Logger* ptr_logger = nullptr, //!< output file stream for logging
#ifdef __MPI
               MPI_Comm comm = MPI_COMM_WORLD   //!< MPI comm
#else
               MPI_Comm comm = 0
#endif
    );
    

  private:
    
    //! Read beta projectors from a pseudopotential file of UPF 1.0.0 format
    void read_beta_upf100(std::ifstream& ifs,               //!< input file stream from orbital file
                          const ModuleBase::Logger* ptr_logger = nullptr, //!< output file stream for logging
#ifdef __MPI
                          MPI_Comm comm = MPI_COMM_WORLD    //!< MPI rank
#else
                          MPI_Comm comm = 0
#endif
    );

    //! Read beta projectors from a pseudopotential file of UPF 2.0.1 format
    void read_beta_upf201(std::ifstream& ifs,               //!< input file stream from orbital file
                          bool lspinorb,
                          const ModuleBase::Logger* ptr_logger = nullptr, //!< output file stream for logging
#ifdef __MPI
                          MPI_Comm comm = MPI_COMM_WORLD    //!< MPI rank
#else
                          MPI_Comm comm = 0
#endif
    );

    //! extract the substring between a pair of quotation marks (for UPF v2.0.1)
    std::string trim201(std::string const& str);

    /// extract value string from a string of the form keyword=" value"
    std::string extract201(std::string const& str, std::string const& keyword);

    ModuleBase::SphericalBesselTransformer sbt_;
    
};

#endif
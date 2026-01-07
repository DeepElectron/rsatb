/*
    This file implement the operators that the 2-center integrals compute, includeing:
    1. The overlap operator
    2. The position operator
    3. The velocity operator

    More can be computed, such as:
    1. The non-local pesudopotential operator
    2. The kinetic operator
    ...

    The operator will be computed with a real space format, which gives a index - subblock relation:
    For example: 
        overlap, position, kinetic: IJR -> block
        velocity: IKJRR' -> block
*/
#include "source_basis/module_nao/two_center_bundle.h"
#include "source_cell/atom_pseudo.h"
#include "source_estate/read_pseudo.h"
#include "source_base/module_container/ATen/tensor.h"
#include <vector>
#include <complex>
#ifdef __MPI
    #include <mpi.h>
#endif

class Op2c
{
private:
    TwoCenterBundle tcbd;
    std::vector<Atom_pseudo> psds;
    container::Tensor orb_map{container::DataType::DT_INT, container::TensorShape({0})};
    container::Tensor beta_map{container::DataType::DT_INT, container::TensorShape({0})};
    MPI_Comm comm;
    int rank;
    int nspin;
    bool lspinorb;
    
public:
    Op2c(
        size_t ntype, int nspin, bool lspinorb,
        const std::string& orb_dir, const std::vector<std::string> orb_name, const std::string& psd_dir, const std::vector<std::string> psd_name,
        MPI_Comm comm, const std::string& log_file
    );
    ~Op2c() = default;

    void overlap(size_t itype, size_t jtype, ModuleBase::Vector3<double> Rij, bool is_transpose, std::vector<double>& v, std::vector<double>* dvx = nullptr, std::vector<double>* dvy = nullptr, std::vector<double>* dvz = nullptr);

    void overlap_position(
        size_t itype, size_t jtype, 
        ModuleBase::Vector3<double> Ri, ModuleBase::Vector3<double> Rj, 
        bool is_transpose,
        std::vector<double>& v, std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz
    );

    void orb_r_beta(
        std::vector<size_t>& itype, size_t ktype, 
        std::vector<ModuleBase::Vector3<double>> Ri, ModuleBase::Vector3<double> Rk,
        bool is_transpose,
        std::vector<ModuleBase::matrix>& ob, std::vector<ModuleBase::matrix>& oxb, 
        std::vector<ModuleBase::matrix>& oyb, std::vector<ModuleBase::matrix>& ozb
    );

    void ncomm_IKJ(
        size_t itype, size_t idx, size_t ktype, size_t jtype, size_t jdx, 
        std::vector<ModuleBase::matrix>& ob, std::vector<ModuleBase::matrix>& oxb, 
        std::vector<ModuleBase::matrix>& oyb, std::vector<ModuleBase::matrix>& ozb,
        int npol, bool is_transpose,
        std::vector<std::complex<double>>& vx, std::vector<std::complex<double>>& vy, std::vector<std::complex<double>>& vz
    );

    double get_orb_rcut_max(int itype) const;
    double get_beta_rcut_max(int itype) const;
};
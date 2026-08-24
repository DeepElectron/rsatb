#ifndef RSATB_MODEL_LCAO_OP2C_ENGINE_HPP
#define RSATB_MODEL_LCAO_OP2C_ENGINE_HPP

// The op2c two-centre integral engine, as LCAO uses it.
//
// This is LCAO's own machinery and nothing else's: it tabulates the numerical
// atomic orbitals and the pseudopotential projectors once, and evaluates the
// overlap, the position operator and the velocity from them. A Slater-Koster
// model computes those from parameters instead, and a file-backed model does
// not have them at all -- so neither includes this header, and neither pulls in
// op2c.
//
// It was the op2c half of a class called OperatorConstruction that also held a
// Hamiltonian-file reader and a bag of block-assembly helpers. Those are now
// model/hamiltonian_images.hpp and operator/block_assembly.hpp; what is left is
// named for what it is.
//
// Units: op2c works in Bohr, AtomicData in Angstrom. Every cutoff crossing that
// boundary is converted here, at the boundary.

#include "units.hpp"

#include "../../operator/block_assembly.hpp"

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/image_container.hpp"
#include "vbcsr/core/block_csr.hpp"
#include "../../extern/op2c/src/int2c/op2c.hpp"

#include <complex>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vbcsr;
using namespace vbcsr::atomic;

namespace rsatb {
namespace model {
namespace lcao {

class Op2cEngine {
public:
    std::unique_ptr<Op2c> op2c_;
    MPI_Comm comm;
    int rank;

    Op2cEngine(MPI_Comm c = MPI_COMM_WORLD) : comm(c) {
        MPI_Comm_rank(comm, &rank);
    }
    
    // op2c joins directory and filename with a bare concatenation and, for
    // orbital files only, reports a failed open on stdout without throwing --
    // the unbuilt object is then dereferenced downstream and the process
    // segfaults. Normalise the separator and fail loudly here instead.
    static std::string join_path(const std::string& dir, const std::string& name) {
        if (dir.empty() || dir.back() == '/') {
            return dir + name;
        }
        return dir + "/" + name;
    }

    static void require_readable(const std::string& path, const char* what) {
        std::ifstream probe(path);
        if (!probe.is_open()) {
            throw std::runtime_error(
                std::string("Op2cEngine::init: cannot open ") + what + " file: " + path);
        }
    }

    void init(size_t ntype, int nspin, bool lspinorb,
              const std::string& orb_dir, const std::vector<std::string>& orb_name,
              const std::string& psd_dir, const std::vector<std::string>& psd_name,
              const std::string& log_file) {
        // Every rank checks the same shared filesystem, so a missing file throws
        // collectively rather than deadlocking a subset of ranks.
        std::string orb_root = orb_dir;
        if (!orb_root.empty() && orb_root.back() != '/') {
            orb_root.push_back('/');
        }
        std::string psd_root = psd_dir;
        if (!psd_root.empty() && psd_root.back() != '/') {
            psd_root.push_back('/');
        }

        for (const auto& name : orb_name) {
            if (!name.empty()) {
                require_readable(join_path(orb_root, name), "orbital");
            }
        }
        for (const auto& name : psd_name) {
            if (!name.empty()) {
                require_readable(join_path(psd_root, name), "pseudopotential");
            }
        }

        // Two-center table construction can also throw from op2c (e.g. the radial
        // aliasing guard); surface which inputs were involved.
        //
        // overlap_only=true: rsatb never calls Op2c::kinetic, so skip tabulating
        // the kinetic table (roughly a third of the two-center table memory and
        // build time). pm_build/orb_kcut keep their defaults.
        try {
            op2c_ = std::unique_ptr<Op2c>(new Op2c(ntype, nspin, lspinorb, orb_root, orb_name, psd_root, psd_name, comm, log_file,
                                                   /*pm_build=*/true, /*orb_kcut=*/0.0, /*overlap_only=*/true));
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "Op2cEngine::init: failed to build op2c two-center tables from orb_dir='"
                + orb_root + "', psd_dir='" + psd_root + "': " + e.what());
        }
    }

    bool initialized() const {
        return static_cast<bool>(op2c_);
    }

    bool has_beta_projectors() const {
        return op2c_ && op2c_->tcbd.beta_;
    }

    /// Returns the loaded orbital basis, or nullptr before init().
    ///
    /// Exposed so a model can read its own basis size and cutoff radii out of
    /// the orbital files rather than having the caller declare them.
    const RadialCollection* orbital_radials() const {
        return op2c_ ? op2c_->tcbd.orb_.get() : nullptr;
    }

    /// Returns per-type orbital cutoff radii in Angstrom.
    ///
    /// op2c stores radial grids in Bohr while AtomicData works in Angstrom, so
    /// the conversion belongs here. Without it the three-body velocity graph
    /// was built with cutoffs inflated by 1/kAngstromPerBohr, which produced a
    /// strict superset of the required triples: correct results, but
    /// substantially more work than necessary.
    std::vector<double> orbital_cutoffs(const AtomicData& ad) const {
        if (!op2c_) {
            throw std::runtime_error("Op2cEngine is not initialized");
        }

        const int ntype = ad.n_type();
        std::vector<double> cutoffs(ntype, 0.0);
        for (int i = 0; i < ntype; ++i) {
            cutoffs[i] = op2c_->get_orb_rcut_max(i) * kAngstromPerBohr;
        }
        return cutoffs;
    }

    /// Returns per-type beta-projector cutoff radii in Angstrom.
    std::vector<double> projector_cutoffs(const AtomicData& ad) const {
        if (!op2c_) {
            throw std::runtime_error("Op2cEngine is not initialized");
        }
        if (!has_beta_projectors()) {
            throw std::runtime_error("Velocity construction requires beta projectors");
        }

        const int ntype = ad.n_type();
        std::vector<double> cutoffs(ntype, 0.0);
        for (int i = 0; i < ntype; ++i) {
            cutoffs[i] = op2c_->get_beta_rcut_max(i) * kAngstromPerBohr;
        }
        return cutoffs;
    }

    DistGraph* velocity_graph(AtomicData& ad) const {
        auto orbrmax = orbital_cutoffs(ad);
        auto psdrmax = projector_cutoffs(ad);
        return ad.get_graph3b(orbrmax, psdrmax);
    }

    // Wrappers for Op2c setup (if needed, or access op2c_ directly)
    // Op2c constructor handles setup.

    /// Builds the real-space Hamiltonian images from a precomputed file.
    ///
    /// The reader abstracts the on-disk layout, so this loop is identical for
    /// every supported HamiltonianFormat. A block the file does not contain is
    /// simply absent from the image; that is expected for pairs beyond the
    /// producer's own cutoff, which can be shorter than the graph cutoff.
    ///
    /// Args:
    ///   ad: Partitioned atomic data supplying the block structure.
    ///   reader: Open reader for the Hamiltonian file.
    std::unique_ptr<ImageContainer<double>> build_overlap(AtomicData& ad) {
        
        auto img = std::make_unique<ImageContainer<double>>(&ad);
        
        int ntype = ad.n_type();
        int nstart, nend;
        ad.get_vertex_range_local(&nstart, &nend);

        // An exception must not leave an OpenMP structured block, so a failing
        // add_block (an R absent from the AtomicData edge set) is recorded here
        // and rethrown once the region closes, as in BuildHamiltonianImagesFromFile.
        std::string failure;
        auto record_failure = [&failure](const std::string& message) {
            #pragma omp critical(rsatb_op2c_engine_failure)
            {
                if (failure.empty()) failure = message;
            }
        };

        // Parallel over atoms, as in BuildHamiltonianImagesFromFile. Everything
        // op2c touches on this path is const (TwoCenterIntegrator::calculate ->
        // TwoCenterTable::lookup, no mutable members), and BlockSpMat::add_block
        // keeps per-thread remote buffers, so distinct rows never collide.
        // vR is declared per iteration rather than hoisted: it is three doubles,
        // so there is no buffer worth reusing, and sharing it was the data race
        // that got this directive commented out in the first place.
        #pragma omp parallel for
        for(int i=nstart; i<nend; ++i) {
            if (ad.atom_is_ghost(i)) continue;

            const ModuleBase::Vector3<double> vR(0, 0, 0);

            int inorb;
            ad.get_atom_norb(i, &inorb);
            int itype;
            ad.get_atom_type(i, &itype);

            // On-site
            std::vector<double> vS(inorb*inorb);
            
            op2c_->overlap(itype, itype, vR, false, vS, nullptr, nullptr, nullptr);
            
            // Set blocks
            int global_i = ad.get_global_index(i);
            std::vector<int> R = {0, 0, 0};
            try {
                img->add_block(R, global_i, global_i, vS.data(), inorb, inorb, AssemblyMode::ADD, MatrixLayout::RowMajor);
            } catch (const std::exception& error) {
                record_failure(error.what());
            }
        }
        
        // Edges
        int estart, eend;
        ad.get_edge_range_local(&estart, &eend);
        const double* cell = ad.cell_ptr();

        // Parallel over edges: each edge owns a distinct (i, j, R) block, so the
        // ADD accumulations never target the same destination.
        #pragma omp parallel for
        for(int ie=estart; ie<eend; ++ie) {
            ModuleBase::Vector3<double> vR;

            int i, j;
            ad.get_connected_atoms(ie, &i, &j);
            int inorb, jnorb;
            ad.get_atom_norb(i, &inorb);
            ad.get_atom_norb(j, &jnorb);
            
            if (inorb > 0 && jnorb > 0) {
                int itype, jtype;
                ad.get_atom_type(i, &itype);
                ad.get_atom_type(j, &jtype);
                
                int Rx_, Ry_, Rz_;
                ad.get_edge_shift_vec(ie, &Rx_, &Ry_, &Rz_);
                
                double rx = Rx_*cell[0] + Ry_*cell[3] + Rz_*cell[6];
                double ry = Rx_*cell[1] + Ry_*cell[4] + Rz_*cell[7];
                double rz = Rx_*cell[2] + Ry_*cell[5] + Rz_*cell[8];
                
                double x, y, z;
                ad.get_pos(j, &x, &y, &z);
                vR.set(rx+x, ry+y, rz+z);
                ad.get_pos(i, &x, &y, &z);
                vR.set(vR.x-x, vR.y-y, vR.z-z);
                
                vR *= 1.8897259886;
                
                std::vector<double> vS(inorb*jnorb);
                
                op2c_->overlap(itype, jtype, vR, false, vS, nullptr, nullptr, nullptr);
                
                // Set blocks
                int global_i = ad.get_global_index(i);
                int global_j = ad.get_global_index(j);
                std::vector<int> R = {Rx_, Ry_, Rz_};
                try {
                    img->add_block(R, global_i, global_j, vS.data(), inorb, jnorb, AssemblyMode::ADD, MatrixLayout::RowMajor);
                } catch (const std::exception& error) {
                    record_failure(error.what());
                }
            }
        }

        if (!failure.empty()) throw std::runtime_error(failure);

        img->assemble();
        return img;
    }

    template <typename MatrixType>
    void build_position(
        MatrixType* X,
        MatrixType* Y,
        MatrixType* Z,
        AtomicData& ad,
        double Kx,
        double Ky,
        double Kz
    ) {
        if (!X && !Y && !Z) {
            return;
        }

        constexpr double bohr_per_angstrom = kBohrPerAngstrom;
        constexpr double angstrom_per_bohr = kAngstromPerBohr;

        int nstart = 0;
        int nend = 0;
        ad.get_vertex_range_local(&nstart, &nend);

        // Every scratch value below is already declared inside the loop and each
        // atom writes only its own row, so this parallelises without changes.
        #pragma omp parallel for
        for (int i = nstart; i < nend; ++i) {
            if (ad.atom_is_ghost(i)) {
                continue;
            }

            int inorb = 0;
            int itype = 0;
            ad.get_atom_norb(i, &inorb);
            ad.get_atom_type(i, &itype);

            double xi = 0.0;
            double yi = 0.0;
            double zi = 0.0;
            ad.get_pos(i, &xi, &yi, &zi);
            ModuleBase::Vector3<double> Ri(xi * bohr_per_angstrom, yi * bohr_per_angstrom, zi * bohr_per_angstrom);

            std::vector<double> vS(inorb * inorb), vX, vY, vZ;
            if (X) vX.resize(inorb * inorb);
            if (Y) vY.resize(inorb * inorb);
            if (Z) vZ.resize(inorb * inorb);

            op2c_->overlap_position(itype, itype, Ri, Ri, false, vS, vX, vY, vZ);

            if (X) op::AddLocalBlock(*X, ad, i, i, 0, 0, 0, Kx, Ky, Kz, vX.data(), angstrom_per_bohr);
            if (Y) op::AddLocalBlock(*Y, ad, i, i, 0, 0, 0, Kx, Ky, Kz, vY.data(), angstrom_per_bohr);
            if (Z) op::AddLocalBlock(*Z, ad, i, i, 0, 0, 0, Kx, Ky, Kz, vZ.data(), angstrom_per_bohr);

            const auto& edges = ad.get_atom_edges(i);
            for (int ie : edges) {
                int j = ad.get_edge_dst(ie);
                int jnorb = 0;
                int jtype = 0;
                ad.get_atom_norb(j, &jnorb);
                ad.get_atom_type(j, &jtype);

                double rx = 0.0;
                double ry = 0.0;
                double rz = 0.0;
                ad.get_edge_vec(ie, &rx, &ry, &rz);
                ModuleBase::Vector3<double> Rj(
                    (xi + rx) * bohr_per_angstrom,
                    (yi + ry) * bohr_per_angstrom,
                    (zi + rz) * bohr_per_angstrom);
                
                double frx = rx, fry = ry, frz = rz;
                ad.invert_cell(&frx, &fry, &frz);

                vS.assign(inorb * jnorb, 0.0);
                if (X) vX.assign(inorb * jnorb, 0.0);
                if (Y) vY.assign(inorb * jnorb, 0.0);
                if (Z) vZ.assign(inorb * jnorb, 0.0);

                op2c_->overlap_position(itype, jtype, Ri, Rj, false, vS, vX, vY, vZ);

                if (X) op::AddLocalBlock(*X, ad, i, j, frx, fry, frz, Kx, Ky, Kz, vX.data(), angstrom_per_bohr);
                if (Y) op::AddLocalBlock(*Y, ad, i, j, frx, fry, frz, Kx, Ky, Kz, vY.data(), angstrom_per_bohr);
                if (Z) op::AddLocalBlock(*Z, ad, i, j, frx, fry, frz, Kx, Ky, Kz, vZ.data(), angstrom_per_bohr);
            }
        }

        if (X) X->assemble();
        if (Y) Y->assemble();
        if (Z) Z->assemble();
    }

    template <typename MatrixType>
    std::unique_ptr<ImageContainer<double>> build_overlap_velocity(MatrixType* Vx, MatrixType* Vy, MatrixType* Vz,
                       AtomicData& ad,
                       double Kx, double Ky, double Kz) {

        // The 3-body term below goes through orb_r_beta/ncomm_IKJ, which need the
        // beta projectors. Without pseudopotentials op2c only prints a message and
        // returns, which would silently drop the non-local commutator and yield a
        // velocity operator that is wrong rather than absent.
        if (!has_beta_projectors()) {
            throw std::runtime_error(
                "build_overlap_velocity requires beta projectors (pseudopotentials); "
                "none are loaded, so the non-local commutator term cannot be built.");
        }

        auto img_S = std::make_unique<ImageContainer<double>>(&ad);

        int nstart, nend;
        ad.get_vertex_range_local(&nstart, &nend);
        
        std::complex<double> IMAG_UNIT(0, 1);
        // erfactor = Ha->eV / (A->Bohr) = 14.399647 = (hbar^2/m_e) x
        // (Bohr^-1 -> A^-1): it turns the gradient integrals (Bohr^-1) into
        // hbar * velocity in eV*Angstrom -- the canonical velocity convention
        // of the codebase (see commutator_velocity.hpp UNITS). Do NOT divide
        // by hbar here: the correlators step time in 1/eV and the SI constant
        // in observables.hpp assumes exactly these units.
        std::complex<double> erfactor = 27.211386245981 / 1.8897259886;

        // As in build_overlap: an add_block failure cannot propagate out of the
        // OpenMP region, so it is recorded and rethrown once the region closes.
        std::string failure;
        auto record_failure = [&failure](const std::string& message) {
            #pragma omp critical(rsatb_op2c_engine_failure)
            {
                if (failure.empty()) failure = message;
            }
        };

        // 1. 2-body part (Overlap + Gradient)
        // Safe to parallelise as written: vR and every scratch vector below are
        // already declared inside the loop, and each atom writes its own row.
        #pragma omp parallel for
        for(int i=nstart; i<nend; ++i) {
            if (ad.atom_is_ghost(i)) continue;
            
            int inorb;
            ad.get_atom_norb(i, &inorb);
            int itype;
            ad.get_atom_type(i, &itype);
            
            double x, y, z;
            ad.get_pos(i, &x, &y, &z);
            ModuleBase::Vector3<double> R2(x, y, z);
            R2 *= 1.8897259886; // Bohr
            ModuleBase::Vector3<double> vR(0,0,0);
            
            // On-site
            std::vector<double> vS(inorb*inorb), vX, vY, vZ;
            if(Vx) vX.resize(inorb*inorb);
            if(Vy) vY.resize(inorb*inorb);
            if(Vz) vZ.resize(inorb*inorb);
            std::vector<double> *pvX = Vx ? &vX : nullptr;
            std::vector<double> *pvY = Vy ? &vY : nullptr;
            std::vector<double> *pvZ = Vz ? &vZ : nullptr;
            
            
            op2c_->overlap(itype, itype, vR, false, vS, pvX, pvY, pvZ);
            
            int global_i = ad.get_global_index(i);
            std::vector<int> R = {0, 0, 0};
            // op2c emits row-major; state it explicitly rather than relying on the
            // library's default layout argument.
            try {
                img_S->add_block(R, global_i, global_i, vS.data(), inorb, inorb,
                                 AssemblyMode::ADD, MatrixLayout::RowMajor);
            } catch (const std::exception& error) {
                record_failure(error.what());
            }

            if(Vx) op::AddLocalBlock(*Vx, ad, i, i, 0, 0, 0, Kx, Ky, Kz, vX.data(), IMAG_UNIT * erfactor);
            if(Vy) op::AddLocalBlock(*Vy, ad, i, i, 0, 0, 0, Kx, Ky, Kz, vY.data(), IMAG_UNIT * erfactor);
            if(Vz) op::AddLocalBlock(*Vz, ad, i, i, 0, 0, 0, Kx, Ky, Kz, vZ.data(), IMAG_UNIT * erfactor);
            
            // Edges
            const auto& edges = ad.get_atom_edges(i);
            for(int ie : edges) {
                int j = ad.get_edge_dst(ie);
                int jnorb;
                ad.get_atom_norb(j, &jnorb);
                int jtype;
                ad.get_atom_type(j, &jtype);
                
                int Rx_, Ry_, Rz_;
                ad.get_edge_shift_vec(ie, &Rx_, &Ry_, &Rz_);
                
                double rx, ry, rz;
                ad.get_edge_vec(ie, &rx, &ry, &rz); // Rj - Ri + Rshift
                
                vR.set(rx, ry, rz);
                vR *= 1.8897259886;
                
                // Re-allocate vectors
                vS.assign(inorb*jnorb, 0);
                if(Vx) vX.assign(inorb*jnorb, 0);
                if(Vy) vY.assign(inorb*jnorb, 0);
                if(Vz) vZ.assign(inorb*jnorb, 0);
                std::vector<double> *pvX = Vx ? &vX : nullptr;
                std::vector<double> *pvY = Vy ? &vY : nullptr;
                std::vector<double> *pvZ = Vz ? &vZ : nullptr;
                
                op2c_->overlap(itype, jtype, vR, false, vS, pvX, pvY, pvZ);
                
                int global_j = ad.get_global_index(j);
                std::vector<int> R_vec = {Rx_, Ry_, Rz_};
                try {
                    img_S->add_block(R_vec, global_i, global_j, vS.data(), inorb, jnorb,
                                     AssemblyMode::ADD, MatrixLayout::RowMajor);
                } catch (const std::exception& error) {
                    record_failure(error.what());
                }

                double frx = rx, fry = ry, frz = rz;
                ad.invert_cell(&frx, &fry, &frz);
                
                if(Vx) op::AddLocalBlock(*Vx, ad, i, j, frx, fry, frz, Kx, Ky, Kz, vX.data(), IMAG_UNIT * erfactor);
                if(Vy) op::AddLocalBlock(*Vy, ad, i, j, frx, fry, frz, Kx, Ky, Kz, vY.data(), IMAG_UNIT * erfactor);
                if(Vz) op::AddLocalBlock(*Vz, ad, i, j, frx, fry, frz, Kx, Ky, Kz, vZ.data(), IMAG_UNIT * erfactor);
            }
        }

        if (!failure.empty()) throw std::runtime_error(failure);

        // 2. 3-body part (v_comm)
        if (Vx || Vy || Vz) {
            for(int k=nstart; k<nend; ++k) {
                if (ad.atom_is_ghost(k)) continue;
                
                int ktype;
                ad.get_atom_type(k, &ktype);
                
                // Neighbors
                const auto& edges = ad.get_atom_edges(k);
                int nei = edges.size();
                
                std::vector<size_t> conn_type(nei+1);
                std::vector<int> conn_idx(nei+1);
                std::vector<ModuleBase::Vector3<double>> Ri(nei+1);
                
                int in = 0;
                for(int ie : edges) {
                    int idx = ad.get_edge_dst(ie); // neighbor index (local)
                    int tp;
                    ad.get_atom_type(idx, &tp);
                    
                    double rx, ry, rz;
                    ad.get_edge_vec(ie, &rx, &ry, &rz); // Ri - Rk
                    
                    conn_type[in] = tp;
                    conn_idx[in] = idx;
                    Ri[in].set(rx, ry, rz); // Relative to Rk (at 0)
                    Ri[in] *= 1.8897259886; // Bohr
                    in++;
                }
                
                // On-site (k itself)
                conn_type[nei] = ktype;
                conn_idx[nei] = k;
                Ri[nei].set(0, 0, 0);
                
                // Call orb_r_beta
                std::vector<ModuleBase::matrix> ob(nei+1), oxb(nei+1), oyb(nei+1), ozb(nei+1);
                ModuleBase::Vector3<double> Rk_vec(0,0,0); // Center is at 0 relative to itself
                
                op2c_->orb_r_beta(conn_type, ktype, Ri, Rk_vec, false, ob, oxb, oyb, ozb);
                
                // Double loop
                for(int in=0; in<conn_type.size(); ++in) {
                    int itype = conn_type[in];
                    int idx = conn_idx[in]; // local index of i
                    int inorb;
                    ad.get_atom_norb(idx, &inorb);
                    
                    for(int jn=0; jn<conn_type.size(); ++jn) {
                        int jtype = conn_type[jn];
                        int jdx = conn_idx[jn]; // local index of j
                        int jnorb;
                        ad.get_atom_norb(jdx, &jnorb);
                        
                        // Cutoff check
                        double rki = Ri[in].norm(); // Bohr
                        double rkj = Ri[jn].norm(); // Bohr
                        
                        double cutoff_ki = op2c_->get_beta_rcut_max(ktype) + op2c_->get_orb_rcut_max(itype);
                        double cutoff_kj = op2c_->get_beta_rcut_max(ktype) + op2c_->get_orb_rcut_max(jtype);
                        
                        if(rki > cutoff_ki || rkj > cutoff_kj) continue;
                        
                        // Compute dR = Rj - Ri
                        ModuleBase::Vector3<double> dR = Ri[jn] - Ri[in];
                        dR /= 1.8897259886;
                        
                        // Invert cell for phase
                        double drx = dR.x, dry = dR.y, drz = dR.z;
                        ad.invert_cell(&drx, &dry, &drz);
                        
                        std::vector<std::complex<double>> vX(inorb*jnorb), vY(inorb*jnorb), vZ(inorb*jnorb);
                        
                        op2c_->ncomm_IKJ(itype, in, ktype, jtype, jn, 
                                       ob, oxb, oyb, ozb, 
                                       1, false, vX, vY, vZ);
                                       
                        // Add to matrices
                        int i_gid = ad.get_global_index(idx);
                        int j_gid = ad.get_global_index(jdx);
                        
                        if(Vx) op::AddGlobalBlock(*Vx, i_gid, j_gid, inorb, jnorb, drx, dry, drz, Kx, Ky, Kz, vX.data(), erfactor);
                        if(Vy) op::AddGlobalBlock(*Vy, i_gid, j_gid, inorb, jnorb, drx, dry, drz, Kx, Ky, Kz, vY.data(), erfactor);
                        if(Vz) op::AddGlobalBlock(*Vz, i_gid, j_gid, inorb, jnorb, drx, dry, drz, Kx, Ky, Kz, vZ.data(), erfactor);
                    }
                }
            }
        }
        
        img_S->assemble();
        
        if(Vx) Vx->assemble();
        if(Vy) Vy->assemble();
        if(Vz) Vz->assemble();
        
        return img_S;
    }


};

}  // namespace lcao
}  // namespace model
}  // namespace rsatb

#endif  // RSATB_MODEL_LCAO_OP2C_ENGINE_HPP

#ifndef RSATB_BACKEND_OPERATOR_CONSTRUCTION_HPP
#define RSATB_BACKEND_OPERATOR_CONSTRUCTION_HPP

#include "vbcsr/core/atomic/atomic_data.hpp"
#include "vbcsr/core/atomic/image_container.hpp"
#include "vbcsr/core/block_csr.hpp"
#include "../io/hdf5.hpp"
#include "../extern/op2c/src/int2c/op2c.hpp"
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

using namespace vbcsr;
using namespace vbcsr::atomic;

namespace rsatb {
namespace backend {

/*
 * For periodic operator algebra we often need the first lattice moment of a
 * real-space image decomposition:
 *
 *   M_alpha = sum_R R_alpha A(R),
 *
 * where A(R) is the operator image associated with lattice translation R and
 * R_alpha is its Cartesian component along axis alpha.
 *
 * This object stores that first moment independently from any position
 * representation. The diagonal orthogonal-basis position term
 *
 *   d_alpha(i, mu) = r_i,alpha
 *
 * belongs to the position state and is built separately.
 */
struct FirstLatticeMomentAxisData {
    using Scalar = std::complex<double>;
    using Matrix = BlockSpMat<Scalar>;

    std::unique_ptr<Matrix> first_moment;
};

struct FirstLatticeMomentTriplet {
    FirstLatticeMomentAxisData x;
    FirstLatticeMomentAxisData y;
    FirstLatticeMomentAxisData z;
};

namespace detail {

// Convert the integer image label R = (n1, n2, n3) to its Cartesian component
// along the selected axis:
//
//   R_alpha = n1 * a1_alpha + n2 * a2_alpha + n3 * a3_alpha.
inline double cell_shift_component(const AtomicData& ad, const std::vector<int>& shift, int axis_idx) {
    const auto& cell = ad.cell;
    return static_cast<double>(shift[0]) * cell[axis_idx]
         + static_cast<double>(shift[1]) * cell[axis_idx + 3]
         + static_cast<double>(shift[2]) * cell[axis_idx + 6];
}

// Build the first image moment along axis alpha:
//
//   M_alpha = sum_R R_alpha A(R),
//
// where each real-space image matrix A(R) is weighted by the Cartesian shift of
// that image. In the present backend A(R) is the Hamiltonian image container.
inline std::unique_ptr<FirstLatticeMomentAxisData::Matrix> make_first_moment_matrix(
    AtomicData& ad,
    ImageContainer<double>& images,
    int axis_idx
) {
    auto* graph = ad.graph;
    auto result = std::unique_ptr<FirstLatticeMomentAxisData::Matrix>(
        images.accumulate_weighted_images<FirstLatticeMomentAxisData::Scalar>(
            graph,
            [&](const std::vector<int>& shift) {
                return cell_shift_component(ad, shift, axis_idx);
            }));
    result->assemble();
    return result;
}

// Build the orthogonal-basis diagonal position vector by replicating the atomic
// coordinate of each owned atom over all of its orbitals:
//
//   d_alpha(i, mu) = r_i,alpha.
inline DistVector<std::complex<double>> make_orthogonal_diagonal_coordinate_vector(
    AtomicData& ad,
    int axis_idx
) {
    auto* graph = ad.graph;
    DistVector<std::complex<double>> diagonal(graph);
    diagonal.set_constant(std::complex<double>(0.0, 0.0));

    const int n_local_blocks = static_cast<int>(graph->block_sizes.size());
    for (int local_block = 0; local_block < n_local_blocks; ++local_block) {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        ad.get_pos(local_block, &x, &y, &z);
        const double coord = (axis_idx == 0) ? x : ((axis_idx == 1) ? y : z);
        const int norb = graph->block_sizes[local_block];
        const int offset = graph->block_offsets[local_block];
        for (int orb = 0; orb < norb; ++orb) {
            diagonal[offset + orb] = std::complex<double>(coord, 0.0);
        }
    }

    return diagonal;
}

} // namespace detail

// Build the first lattice moment along x/y/z. An axis is active iff at least
// one image carries a nonzero lattice translation along that axis.
inline std::unique_ptr<FirstLatticeMomentTriplet> build_first_lattice_moment_triplet(
    AtomicData& atomic_data,
    ImageContainer<double>& hamiltonian_images
) {
    if (hamiltonian_images.atom_data != &atomic_data || hamiltonian_images.base_graph != atomic_data.graph) {
        throw std::runtime_error(
            "First-lattice-moment construction requires Hamiltonian images built from the same AtomicData graph.");
    }

    auto triplet = std::make_unique<FirstLatticeMomentTriplet>();
    FirstLatticeMomentAxisData* axes[3] = {&triplet->x, &triplet->y, &triplet->z};

    for (int axis_idx = 0; axis_idx < 3; ++axis_idx) {
        bool has_lattice_shift = false;
        for (const auto& entry : hamiltonian_images.image_blocks) {
            if (std::abs(detail::cell_shift_component(atomic_data, entry.first, axis_idx)) > 1e-12) {
                has_lattice_shift = true;
                break;
            }
        }

        if (has_lattice_shift) {
            axes[axis_idx]->first_moment =
                detail::make_first_moment_matrix(atomic_data, hamiltonian_images, axis_idx);
        }
    }

    return triplet;
}

class OperatorConstruction {
public:
    std::unique_ptr<Op2c> op2c_;
    MPI_Comm comm;
    int rank;

    OperatorConstruction(MPI_Comm c = MPI_COMM_WORLD) : comm(c) {
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
                std::string("OperatorConstruction::init: cannot open ") + what + " file: " + path);
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
                "OperatorConstruction::init: failed to build op2c two-center tables from orb_dir='"
                + orb_root + "', psd_dir='" + psd_root + "': " + e.what());
        }
    }

    bool initialized() const {
        return static_cast<bool>(op2c_);
    }

    bool has_beta_projectors() const {
        return op2c_ && op2c_->tcbd.beta_;
    }

    std::vector<double> orbital_cutoffs(const AtomicData& ad) const {
        if (!op2c_) {
            throw std::runtime_error("OperatorConstruction is not initialized");
        }

        const int ntype = ad.n_type();
        std::vector<double> cutoffs(ntype, 0.0);
        for (int i = 0; i < ntype; ++i) {
            cutoffs[i] = op2c_->get_orb_rcut_max(i);
        }
        return cutoffs;
    }

    std::vector<double> projector_cutoffs(const AtomicData& ad) const {
        if (!op2c_) {
            throw std::runtime_error("OperatorConstruction is not initialized");
        }
        if (!has_beta_projectors()) {
            throw std::runtime_error("Velocity construction requires beta projectors");
        }

        const int ntype = ad.n_type();
        std::vector<double> cutoffs(ntype, 0.0);
        for (int i = 0; i < ntype; ++i) {
            cutoffs[i] = op2c_->get_beta_rcut_max(i);
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

    // Build matrix from HDF5
    // Reads blocks from HDF5 and fills BlockSpMat
    // Build matrix from HDF5
    // Reads blocks from HDF5 and fills ImageContainer
    std::unique_ptr<ImageContainer<double>> build_from_hdf5(AtomicData& ad, rsatb::io::HDF5Handler& h5) {
        auto img = std::make_unique<ImageContainer<double>>(&ad);
        
        int nstart, nend;
        ad.get_vertex_range_local(&nstart, &nend);
        bool found;
        
        // Loop over atoms
        #pragma omp parallel for
        for(int i=nstart; i<nend; ++i) {
            if (ad.atom_is_ghost(i)) continue;
            
            int inorb;
            ad.get_atom_norb(i, &inorb);
            int atom_idx;
            ad.get_atom_id(i, &atom_idx); // Original index
            
            // Read On-site
            double* v = nullptr;
            h5.read_from_ijR_dense(atom_idx, atom_idx, 0, 0, 0, &inorb, &inorb, &v, &found);
            if (found) {
                int global_i = ad.get_global_index(i);
                std::vector<int> R = {0, 0, 0};
                img->add_block(R, global_i, global_i, v, inorb, inorb, AssemblyMode::ADD, MatrixLayout::RowMajor);
                delete[] v;
            }
        }
        
        // Loop over edges
        int estart, eend;
        ad.get_edge_range_local(&estart, &eend);
        #pragma omp parallel for
        for(int ie=estart; ie<eend; ++ie) {
            int i, j;
            ad.get_connected_atoms(ie, &i, &j);
            
            int inorb, jnorb;
            ad.get_atom_norb(i, &inorb);
            ad.get_atom_norb(j, &jnorb);
            
            if (inorb > 0 && jnorb > 0) {
                int Rx, Ry, Rz;
                ad.get_edge_shift_vec(ie, &Rx, &Ry, &Rz);
                
                int atom_i;
                ad.get_atom_id(i, &atom_i);
                int atom_j;
                ad.get_atom_id(j, &atom_j);
                
                int m, n;
                double* buffer = nullptr;
                bool found = false;
                
                h5.read_from_ijR_dense(atom_i, atom_j, Rx, Ry, Rz, &m, &n, &buffer, &found);
                
                if (found) {
                    if (m != inorb || n != jnorb) {
                        std::cerr << "Dimension mismatch in HDF5 read: expected " << inorb << "x" << jnorb << ", got " << m << "x" << n << std::endl;
                    } else {
                        int global_i = ad.get_global_index(i);
                        int global_j = ad.get_global_index(j);
                        std::vector<int> R = {Rx, Ry, Rz};
                        img->add_block(R, global_i, global_j, buffer, inorb, jnorb, AssemblyMode::ADD, MatrixLayout::RowMajor);
                    }
                    if(buffer) delete[] buffer;
                }
            }
        }
        
        img->assemble();
        return img;
    }

    std::unique_ptr<ImageContainer<double>> build_overlap(AtomicData& ad) {
        
        auto img = std::make_unique<ImageContainer<double>>(&ad);
        
        int ntype = ad.n_type();
        int nstart, nend;
        ad.get_vertex_range_local(&nstart, &nend);
        
        ModuleBase::Vector3<double> vR(0,0,0);
        // #pragma omp parallel for
        for(int i=nstart; i<nend; ++i) {
            if (ad.atom_is_ghost(i)) continue;
            
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
            img->add_block(R, global_i, global_i, vS.data(), inorb, inorb, AssemblyMode::ADD, MatrixLayout::RowMajor);
        }
        
        // Edges
        int estart, eend;
        ad.get_edge_range_local(&estart, &eend);
        const double* cell = ad.cell_ptr();

        // #pragma omp parallel for
        for(int ie=estart; ie<eend; ++ie) {
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
                img->add_block(R, global_i, global_j, vS.data(), inorb, jnorb, AssemblyMode::ADD, MatrixLayout::RowMajor);
            }
        }
        
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

        constexpr double bohr_per_angstrom = 1.8897259886;
        constexpr double angstrom_per_bohr = 1.0 / bohr_per_angstrom;

        int nstart = 0;
        int nend = 0;
        ad.get_vertex_range_local(&nstart, &nend);

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

            if (X) set_local_block(*X, ad, i, i, 0, 0, 0, Kx, Ky, Kz, vX.data(), angstrom_per_bohr);
            if (Y) set_local_block(*Y, ad, i, i, 0, 0, 0, Kx, Ky, Kz, vY.data(), angstrom_per_bohr);
            if (Z) set_local_block(*Z, ad, i, i, 0, 0, 0, Kx, Ky, Kz, vZ.data(), angstrom_per_bohr);

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

                if (X) set_local_block(*X, ad, i, j, frx, fry, frz, Kx, Ky, Kz, vX.data(), angstrom_per_bohr);
                if (Y) set_local_block(*Y, ad, i, j, frx, fry, frz, Kx, Ky, Kz, vY.data(), angstrom_per_bohr);
                if (Z) set_local_block(*Z, ad, i, j, frx, fry, frz, Kx, Ky, Kz, vZ.data(), angstrom_per_bohr);
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
        std::complex<double> erfactor = 27.211386245981 / 1.8897259886;

        // 1. 2-body part (Overlap + Gradient)
        // #pragma omp parallel for
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
            img_S->add_block(R, global_i, global_i, vS.data(), inorb, inorb,
                             AssemblyMode::ADD, MatrixLayout::RowMajor);
            
            if(Vx) set_local_block(*Vx, ad, i, i, 0, 0, 0, Kx, Ky, Kz, vX.data(), IMAG_UNIT * erfactor);
            if(Vy) set_local_block(*Vy, ad, i, i, 0, 0, 0, Kx, Ky, Kz, vY.data(), IMAG_UNIT * erfactor);
            if(Vz) set_local_block(*Vz, ad, i, i, 0, 0, 0, Kx, Ky, Kz, vZ.data(), IMAG_UNIT * erfactor);
            
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
                img_S->add_block(R_vec, global_i, global_j, vS.data(), inorb, jnorb,
                                 AssemblyMode::ADD, MatrixLayout::RowMajor);

                double frx = rx, fry = ry, frz = rz;
                ad.invert_cell(&frx, &fry, &frz);
                
                if(Vx) set_local_block(*Vx, ad, i, j, frx, fry, frz, Kx, Ky, Kz, vX.data(), IMAG_UNIT * erfactor);
                if(Vy) set_local_block(*Vy, ad, i, j, frx, fry, frz, Kx, Ky, Kz, vY.data(), IMAG_UNIT * erfactor);
                if(Vz) set_local_block(*Vz, ad, i, j, frx, fry, frz, Kx, Ky, Kz, vZ.data(), IMAG_UNIT * erfactor);
            }
        }
        
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
                        
                        if(Vx) set_global_block(*Vx, ad, i_gid, j_gid, inorb, jnorb, drx, dry, drz, Kx, Ky, Kz, vX.data(), erfactor);
                        if(Vy) set_global_block(*Vy, ad, i_gid, j_gid, inorb, jnorb, drx, dry, drz, Kx, Ky, Kz, vY.data(), erfactor);
                        if(Vz) set_global_block(*Vz, ad, i_gid, j_gid, inorb, jnorb, drx, dry, drz, Kx, Ky, Kz, vZ.data(), erfactor);
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

private:
    void cal_phase(double Rx, double Ry, double Rz, double Kx, double Ky, double Kz, std::complex<double> *phase) {
        *phase = std::complex<double>(0.0, -2.0 * M_PI * ((Rx * Kx) + (Ry * Ky) + (Rz * Kz)));
        *phase = std::exp(*phase);
    }

    template <typename MatrixType>
    void set_local_block(MatrixType& mat, AtomicData& ad, int i, int j, 
                        int Rx, int Ry, int Rz, 
                        double Kx, double Ky, double Kz, 
                        const double* v, std::complex<double> factor = 1.0) {
        int inorb, jnorb;
        ad.get_atom_norb(i, &inorb);
        ad.get_atom_norb(j, &jnorb);
        
        std::complex<double> phase;
        cal_phase(Rx, Ry, Rz, Kx, Ky, Kz, &phase);
        phase *= factor;
        
        std::vector<std::complex<double>> v_complex(inorb * jnorb);
        for(int k=0; k<inorb*jnorb; ++k) {
            v_complex[k] = v[k] * phase;
        }
        
        mat.update_local_block(i, j, v_complex.data(), inorb, jnorb, AssemblyMode::ADD, MatrixLayout::RowMajor);
    }

    template <typename MatrixType>
    void set_local_block(MatrixType& mat, AtomicData& ad, int i, int j, 
                        int Rx, int Ry, int Rz, 
                        double Kx, double Ky, double Kz, 
                        const std::complex<double>* v, std::complex<double> factor = 1.0) {
        int inorb, jnorb;
        ad.get_atom_norb(i, &inorb);
        ad.get_atom_norb(j, &jnorb);
        
        std::complex<double> phase;
        cal_phase(Rx, Ry, Rz, Kx, Ky, Kz, &phase);
        phase *= factor;
        
        std::vector<std::complex<double>> v_complex(inorb * jnorb);
        for(int k=0; k<inorb*jnorb; ++k) {
            v_complex[k] = v[k] * phase;
        }
        
        mat.update_local_block(i, j, v_complex.data(), inorb, jnorb, AssemblyMode::ADD, MatrixLayout::RowMajor);
    }

    template <typename MatrixType>
    void set_local_block(MatrixType& mat, AtomicData& ad, int i, int j, 
                        double Rx, double Ry, double Rz, 
                        double Kx, double Ky, double Kz, 
                        const double* v, std::complex<double> factor = 1.0) {
        int inorb, jnorb;
        ad.get_atom_norb(i, &inorb);
        ad.get_atom_norb(j, &jnorb);
        
        std::complex<double> phase;
        cal_phase(Rx, Ry, Rz, Kx, Ky, Kz, &phase);
        phase *= factor;
        
        std::vector<std::complex<double>> v_complex(inorb * jnorb);
        for(int k=0; k<inorb*jnorb; ++k) {
            v_complex[k] = v[k] * phase;
        }
        
        mat.update_local_block(i, j, v_complex.data(), inorb, jnorb, AssemblyMode::ADD, MatrixLayout::RowMajor);
    }

    template <typename MatrixType>
    void set_local_block(MatrixType& mat, AtomicData& ad, int i, int j, 
                        double Rx, double Ry, double Rz, 
                        double Kx, double Ky, double Kz, 
                        const std::complex<double>* v, std::complex<double> factor = 1.0) {
        int inorb, jnorb;
        ad.get_atom_norb(i, &inorb);
        ad.get_atom_norb(j, &jnorb);
        
        std::complex<double> phase;
        cal_phase(Rx, Ry, Rz, Kx, Ky, Kz, &phase);
        phase *= factor;
        
        std::vector<std::complex<double>> v_complex(inorb * jnorb);
        for(int k=0; k<inorb*jnorb; ++k) {
            v_complex[k] = v[k] * phase;
        }
        
        mat.update_local_block(i, j, v_complex.data(), inorb, jnorb, AssemblyMode::ADD, MatrixLayout::RowMajor);
    }

    template <typename MatrixType>
    void set_global_block(MatrixType& mat, AtomicData& ad, int i_gid, int j_gid, 
                         int inorb, int jnorb,
                         double dx, double dy, double dz, 
                         double Kx, double Ky, double Kz, 
                         const std::complex<double>* v, std::complex<double> factor = 1.0) {
        
        std::complex<double> phase;
        cal_phase(dx, dy, dz, Kx, Ky, Kz, &phase);
        phase *= factor;
        
        std::vector<std::complex<double>> v_complex(inorb * jnorb);
        for(int k=0; k<inorb*jnorb; ++k) {
            v_complex[k] = v[k] * phase;
        }
        
        mat.add_block(i_gid, j_gid, v_complex.data(), inorb, jnorb, AssemblyMode::ADD, MatrixLayout::RowMajor);
    }
};

} // namespace backend
} // namespace rsatb

#endif

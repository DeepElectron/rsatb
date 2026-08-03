#ifndef RSATB_MODEL_LCAO_MODEL_HPP
#define RSATB_MODEL_LCAO_MODEL_HPP

// LCAO model: a precomputed Hamiltonian plus numerical atomic orbitals.
//
// The Hamiltonian arrives as a format-tagged file. Everything else -- the
// overlap, the position operator, and (given pseudopotentials) the velocity --
// is computed from the orbital basis through op2c.
//
// An LCAO basis is non-orthogonal by construction, so the orbital files are
// required rather than optional. Previously they were inferred from whether
// `orb_names` and `orb_dir` happened to be non-empty, which meant omitting them
// produced a silently orthogonal calculation on top of an LCAO Hamiltonian.
//
// The basis itself is read from those same orbital files, so a caller never
// supplies orbital counts or cutoff radii.

#include "op2c_engine.hpp"

#include "../model.hpp"
#include "../hamiltonian_images.hpp"
#include "../../io/hamiltonian_file.hpp"
#include "../../io/hamiltonian_readers.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsatb {
namespace model {

/// Numerical atomic orbital files, one per atom type in ascending-Z order.
struct OrbitalFiles {
    std::vector<std::string> names;
    std::string dir;

    bool empty() const { return names.empty(); }
};

/// Pseudopotential files, one per atom type in the same order as OrbitalFiles.
struct PseudopotentialFiles {
    std::vector<std::string> names;
    std::string dir;

    bool empty() const { return names.empty(); }
};

struct LcaoConfig {
    io::HamiltonianFile hamiltonian;
    OrbitalFiles orbitals;                    // required: LCAO is non-orthogonal
    PseudopotentialFiles pseudopotentials;    // optional: enables velocity
    SpinSpec spin;
    std::string log_file = "op2c.log";
};

class LcaoModel : public Model {
public:
    /// Builds an LCAO model and derives its basis from the orbital files.
    ///
    /// Args:
    ///   config: Hamiltonian file, orbital files, and optional pseudopotentials.
    ///   comm: Communicator op2c uses to load and tabulate the basis.
    /// Returns:
    ///   A ready model whose basis() reports the orbital counts and cutoff
    ///   radii read from the orbital files.
    /// Throws:
    ///   std::runtime_error if the Hamiltonian file or orbital files are
    ///   missing, or if the pseudopotential count disagrees with the orbital
    ///   count.
    static std::unique_ptr<LcaoModel> Create(LcaoConfig config, MPI_Comm comm) {
        if (config.hamiltonian.empty()) {
            throw std::runtime_error(
                "LCAO setup requires a Hamiltonian file. Set model.hamiltonian.path.");
        }
        if (config.orbitals.empty()) {
            throw std::runtime_error(
                "LCAO setup requires orbital files: an LCAO basis is non-orthogonal, so the "
                "overlap cannot be omitted. Set model.orbitals.names, or use an SKTB/DFTB "
                "model if you intended an orthogonal calculation.");
        }
        if (!config.pseudopotentials.empty() &&
            config.pseudopotentials.names.size() != config.orbitals.names.size()) {
            throw std::runtime_error(
                "LCAO setup expects one pseudopotential file per orbital file (" +
                std::to_string(config.orbitals.names.size()) + " orbital files, " +
                std::to_string(config.pseudopotentials.names.size()) + " pseudopotential files).");
        }

        auto model = std::unique_ptr<LcaoModel>(new LcaoModel(std::move(config), comm));
        model->InitializeIntegrals();
        model->DeriveBasis();
        model->basis_.Validate("LCAO");
        return model;
    }

    std::string name() const override { return "LCAO"; }

    const BasisSpec& basis() const override { return basis_; }

    const SpinSpec& spin() const override { return config_.spin; }

    /// Everything from op2c, except H which is read from the file.
    ///
    /// The basis is non-orthogonal by definition, so the generic route is not
    /// available for any of them: with an overlap neither v nor r is diagonal
    /// in the working basis. Velocity additionally needs the pseudopotential
    /// projectors, so without `.upf` files it is honestly unavailable rather
    /// than silently wrong.
    OperatorSources operator_sources() const override {
        return OperatorSources{
            /*hamiltonian=*/OperatorSource::kModel,
            /*overlap=*/OperatorSource::kModel,
            /*velocity=*/has_velocity_ ? OperatorSource::kModel : OperatorSource::kUnavailable,
            /*position=*/OperatorSource::kModel,
        };
    }

    std::unique_ptr<ImageContainer> BuildHamiltonianImages(AtomicData& atomic_data) const override {
        auto reader = io::OpenHamiltonianFile(config_.hamiltonian);
        return BuildHamiltonianImagesFromFile(atomic_data, *reader);
    }

    std::unique_ptr<ImageContainer> BuildOverlapImages(AtomicData& atomic_data) const override {
        return integrals_->build_overlap(atomic_data);
    }

    std::unique_ptr<VelocityTriplet> BuildVelocity(
        AtomicData& atomic_data,
        const KPoint& kpoint
    ) const override {
        RequireThreeComponentKPoint(kpoint, "LCAO velocity construction");
        if (!has_velocity_) {
            throw std::runtime_error(
                "LCAO velocity requires pseudopotential files with beta projectors. "
                "Set model.pseudopotentials.names.");
        }

        vbcsr::DistGraph* graph = integrals_->velocity_graph(atomic_data);
        auto triplet = std::make_unique<VelocityTriplet>();
        triplet->x = std::make_unique<VelocityTriplet::Matrix>(graph);
        triplet->y = std::make_unique<VelocityTriplet::Matrix>(graph);
        triplet->z = std::make_unique<VelocityTriplet::Matrix>(graph);

        integrals_->build_overlap_velocity(
            triplet->x.get(), triplet->y.get(), triplet->z.get(),
            atomic_data, kpoint[0], kpoint[1], kpoint[2]);
        return triplet;
    }

    std::unique_ptr<PositionTriplet> BuildPosition(
        AtomicData& atomic_data,
        const KPoint& kpoint
    ) const override {
        RequireThreeComponentKPoint(kpoint, "LCAO position construction");

        auto triplet = std::make_unique<PositionTriplet>();
        triplet->x.matrix = std::make_unique<PositionTriplet::Matrix>(atomic_data.get_graph());
        triplet->y.matrix = std::make_unique<PositionTriplet::Matrix>(atomic_data.get_graph());
        triplet->z.matrix = std::make_unique<PositionTriplet::Matrix>(atomic_data.get_graph());

        integrals_->build_position(
            triplet->x.matrix.get(), triplet->y.matrix.get(), triplet->z.matrix.get(),
            atomic_data, kpoint[0], kpoint[1], kpoint[2]);
        return triplet;
    }

    /// Returns the .orb files, resolved against the configured directory.
    ///
    /// These are the radials the real-space contraction evaluates, and they
    /// are the same files op2c built H and S from -- so the per-atom orbital
    /// ordering agrees with the matrix blocks by construction.
    std::vector<std::string> RealSpaceOrbitalFiles() const override {
        std::vector<std::string> paths;
        paths.reserve(config_.orbitals.names.size());
        for (const auto& name : config_.orbitals.names) {
            paths.push_back(lcao::Op2cEngine::join_path(config_.orbitals.dir, name));
        }
        return paths;
    }

private:
    LcaoModel(LcaoConfig config, MPI_Comm comm)
        : config_(std::move(config)), comm_(comm) {}

    // op2c tabulation is expensive, so it happens once here and is reused for
    // every k-point. The previous code rebuilt the two-center tables on every
    // velocity and position call.
    void InitializeIntegrals() {
        integrals_ = std::make_shared<lcao::Op2cEngine>(comm_);
        integrals_->init(
            config_.orbitals.names.size(),
            config_.spin.Channels(),
            config_.spin.HasSpinOrbit(),
            config_.orbitals.dir,
            config_.orbitals.names,
            config_.pseudopotentials.dir,
            config_.pseudopotentials.names,
            config_.log_file);

        has_velocity_ = !config_.pseudopotentials.empty() && integrals_->has_beta_projectors();
    }

    /// Reads species, orbital counts, and cutoff radii out of the orbital files.
    void DeriveBasis() {
        const auto* orbitals = integrals_->orbital_radials();
        if (orbitals == nullptr) {
            throw std::runtime_error("LCAO setup could not load the orbital basis from op2c.");
        }

        const int n_type = static_cast<int>(config_.orbitals.names.size());
        const int spin_components = config_.spin.Components();

        basis_.type_symbols.clear();
        basis_.type_norb.clear();
        basis_.cutoff_radius.clear();
        basis_.type_symbols.reserve(n_type);
        basis_.type_norb.reserve(n_type);
        basis_.cutoff_radius.reserve(n_type);

        for (int type = 0; type < n_type; ++type) {
            basis_.type_symbols.push_back(vbcsr::atomic::CanonicalSymbol(orbitals->symbol(type)));

            int norb = 0;
            for (int l = 0; l <= orbitals->lmax(type); ++l) {
                norb += (2 * l + 1) * orbitals->nzeta(type, l);
            }
            basis_.type_norb.push_back(norb * spin_components);

            // op2c radial grids are in Bohr; AtomicData works in Angstrom.
            basis_.cutoff_radius.push_back(
                orbitals->rcut_max(type) * lcao::kAngstromPerBohr);
        }
    }

    LcaoConfig config_;
    MPI_Comm comm_ = MPI_COMM_WORLD;
    std::shared_ptr<lcao::Op2cEngine> integrals_;
    BasisSpec basis_;
    bool has_velocity_ = false;
};

}  // namespace model
}  // namespace rsatb

#endif  // RSATB_MODEL_LCAO_MODEL_HPP

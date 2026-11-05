#ifndef RSATB_MODEL_TB_FILE_MODEL_HPP
#define RSATB_MODEL_TB_FILE_MODEL_HPP

// Tight-binding model read from a precomputed Hamiltonian file, orthogonal
// unless an overlap file is given too.
//
// This is the fourth combination noted during the interface redesign: a
// file-backed H that is already in an orthogonal basis, which is what a
// DeePTB Slater-Koster export produces. It is not LCAO -- an LCAO basis is
// non-orthogonal and needs orbital files -- and it is not SKTB, which computes
// H from parameters. Giving it its own model keeps each family's story intact
// instead of hanging an optional Hamiltonian file off the other two.
//
// Unlike every other model, the basis must be supplied: the supported file
// formats carry no basis header, so orbital counts cannot be derived. That is
// a property of the format, and `Create` cross-checks what it can against the
// blocks actually present in the file.
//
// A fitted model with an overlap (DeePTB's nnsk and mix, say) has H(R) and S(R)
// but no orbitals, so it has no dipole integrals and so no true position. It
// reads its overlap here and has no velocity, unless the input asks for the
// Mulliken position (operator/nonorthogonal_velocity.hpp says what that costs).

#include "../model.hpp"
#include "../hamiltonian_images.hpp"

#include "../../io/hamiltonian_file.hpp"
#include "../../io/hamiltonian_readers.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rsatb {
namespace model {

struct TightBindingFileConfig {
    io::HamiltonianFile hamiltonian;

    /// Species in ascending atomic number, matching AtomicData's type indexing.
    std::vector<std::string> type_symbols;
    /// Orbitals per type. Required: the file formats carry no basis header.
    std::vector<int> type_norb;
    /// Graph cutoff radii per type, in Angstrom.
    std::vector<double> cutoff_radius;

    SpinSpec spin;

    /// S(R), real over spatial orbitals; empty for an orthogonal basis.
    io::HamiltonianFile overlap;
    /// "mulliken", or empty: with an overlap, no position and no velocity.
    std::string position;
};

class TightBindingFileModel : public Model {
public:
    static std::unique_ptr<TightBindingFileModel> Create(TightBindingFileConfig config, MPI_Comm comm) {
        if (config.hamiltonian.empty()) {
            throw std::runtime_error(
                "Tight-binding file setup requires a Hamiltonian file. Set model.hamiltonian.path.");
        }
        if (io::NeedsBasisLayout(config.hamiltonian.format)) {
            throw std::runtime_error(
                "The '" + io::ToString(config.hamiltonian.format) + "' Hamiltonian format "
                "carries an LCAO Hamiltonian, which is non-orthogonal and in ABACUS's orbital "
                "order; use the 'lcao' model with the orbital files it was computed in.");
        }

        if (!config.overlap.empty() &&
            config.overlap.format != io::HamiltonianFormat::kDeeptbHdf5) {
            throw std::runtime_error(
                "A tight-binding overlap file must be in the 'deeptb' format, not '" +
                io::ToString(config.overlap.format) + "'.");
        }
        if (!config.position.empty() && config.position != "mulliken") {
            throw std::runtime_error(
                "Tight-binding file position must be 'mulliken' or unset, not '" +
                config.position + "'.");
        }
        if (!config.position.empty() && config.overlap.empty()) {
            throw std::runtime_error(
                "The Mulliken position is built from the overlap, and this model has none: "
                "an orthogonal basis takes the diagonal position.");
        }

        BasisSpec basis;
        basis.type_symbols = std::move(config.type_symbols);
        basis.type_norb = std::move(config.type_norb);
        basis.cutoff_radius = std::move(config.cutoff_radius);
        basis.Validate("Tight-binding file");

        auto model = std::unique_ptr<TightBindingFileModel>(
            new TightBindingFileModel(std::move(config.hamiltonian), std::move(config.overlap),
                                      config.position == "mulliken", std::move(basis), config.spin, comm));
        return model;
    }

    std::string name() const override { return "TightBindingFile"; }

    const BasisSpec& basis() const override { return basis_; }

    const SpinSpec& spin() const override { return spin_; }

    /// H from the file; v and r from the generic route.
    ///
    /// Without an overlap file the basis is orthogonal, which is exactly the
    /// condition the commutator velocity and the diagonal position need. So
    /// this model builds one operator and inherits two. With one, there is no
    /// position to dress the velocity with unless the input chose Mulliken's.
    OperatorSources operator_sources() const override {
        if (!overlap_.empty()) {
            return OperatorSources{
                /*hamiltonian=*/OperatorSource::kModel,
                /*overlap=*/OperatorSource::kModel,
                /*velocity=*/mulliken_position_ ? OperatorSource::kGeneric : OperatorSource::kUnavailable,
                /*position=*/mulliken_position_ ? OperatorSource::kMulliken : OperatorSource::kUnavailable,
            };
        }
        return OperatorSources{
            /*hamiltonian=*/OperatorSource::kModel,
            /*overlap=*/OperatorSource::kUnavailable,
            /*velocity=*/OperatorSource::kGeneric,
            /*position=*/OperatorSource::kGeneric,
        };
    }

    HamiltonianImages BuildHamiltonianImages(AtomicData& atomic_data) const override {
        auto reader = io::OpenHamiltonianFile(hamiltonian_, spin_.Components());
        return BuildHamiltonianImagesFromFile(atomic_data, *reader);
    }

    /// Returns S(R) from the overlap file, or nullptr for an orthogonal basis.
    std::unique_ptr<ImageContainer> BuildOverlapImages(AtomicData& atomic_data) const override {
        if (overlap_.empty()) {
            return nullptr;
        }
        auto reader = io::OpenHamiltonianFile(overlap_);
        return BuildOverlapImagesFromFile(atomic_data, *reader, spin_.Components());
    }

private:
    TightBindingFileModel(io::HamiltonianFile hamiltonian, io::HamiltonianFile overlap, bool mulliken_position,
                          BasisSpec basis, SpinSpec spin, MPI_Comm comm)
        : hamiltonian_(std::move(hamiltonian)), overlap_(std::move(overlap)),
          mulliken_position_(mulliken_position), basis_(std::move(basis)), spin_(spin), comm_(comm) {}

    io::HamiltonianFile hamiltonian_;
    io::HamiltonianFile overlap_;
    bool mulliken_position_ = false;
    BasisSpec basis_;
    SpinSpec spin_;
    MPI_Comm comm_ = MPI_COMM_WORLD;
};

}  // namespace model
}  // namespace rsatb

#endif  // RSATB_MODEL_TB_FILE_MODEL_HPP

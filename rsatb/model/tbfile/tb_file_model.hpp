#ifndef RSATB_MODEL_TB_FILE_MODEL_HPP
#define RSATB_MODEL_TB_FILE_MODEL_HPP

// Tight-binding model read from a precomputed Hamiltonian file, with no
// overlap.
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

        BasisSpec basis;
        basis.type_symbols = std::move(config.type_symbols);
        basis.type_norb = std::move(config.type_norb);
        basis.cutoff_radius = std::move(config.cutoff_radius);
        basis.Validate("Tight-binding file");

        auto model = std::unique_ptr<TightBindingFileModel>(
            new TightBindingFileModel(std::move(config.hamiltonian), std::move(basis), config.spin, comm));
        return model;
    }

    std::string name() const override { return "TightBindingFile"; }

    const BasisSpec& basis() const override { return basis_; }

    const SpinSpec& spin() const override { return spin_; }

    /// Returns an orthogonal capability set.
    ///
    /// Velocity and position follow from the Hamiltonian images and the atomic
    /// positions, as for any orthogonal basis.
    /// H from the file; v and r from the generic route.
    ///
    /// The file carries H(R) already in an orthogonal basis -- there is no
    /// overlap to read and none to compute -- and orthogonality is exactly the
    /// condition the commutator velocity and the diagonal position need. So
    /// this model builds one operator and inherits two.
    OperatorSources operator_sources() const override {
        return OperatorSources{
            /*hamiltonian=*/OperatorSource::kModel,
            /*overlap=*/OperatorSource::kUnavailable,
            /*velocity=*/OperatorSource::kGeneric,
            /*position=*/OperatorSource::kGeneric,
        };
    }

    HamiltonianImages BuildHamiltonianImages(AtomicData& atomic_data) const override {
        auto reader = io::OpenHamiltonianFile(hamiltonian_);
        return BuildHamiltonianImagesFromFile(atomic_data, *reader);
    }

    /// Returns nullptr: this model is orthogonal by construction.
    std::unique_ptr<ImageContainer> BuildOverlapImages(AtomicData&) const override {
        return nullptr;
    }

private:
    TightBindingFileModel(io::HamiltonianFile hamiltonian, BasisSpec basis, SpinSpec spin, MPI_Comm comm)
        : hamiltonian_(std::move(hamiltonian)), basis_(std::move(basis)), spin_(spin), comm_(comm) {}

    io::HamiltonianFile hamiltonian_;
    BasisSpec basis_;
    SpinSpec spin_;
    MPI_Comm comm_ = MPI_COMM_WORLD;
};

}  // namespace model
}  // namespace rsatb

#endif  // RSATB_MODEL_TB_FILE_MODEL_HPP

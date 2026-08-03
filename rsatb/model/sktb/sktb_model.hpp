#ifndef RSATB_MODEL_SKTB_SKTB_MODEL_HPP
#define RSATB_MODEL_SKTB_SKTB_MODEL_HPP

// SKTB model: Slater-Koster parameters given as analytic or tabulated radial
// functions.
//
// The basis is declared, so orbital counts follow from it; the cutoff radii
// follow from the radial functions. Neither is supplied by the caller.
//
// SKTB is orthogonal unless overlap parameters are given. That is a real
// physical choice rather than an accident of which fields were filled in, so
// it is reported through capabilities().

#include "sk_table.hpp"
#include "sktb_construction.hpp"
#include "skf_table.hpp"
#include "../model.hpp"

#include "../../operator/types.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace rsatb {
namespace model {

struct SktbConfig {
    /// species symbol -> basis labels, e.g. {"C": {"pz"}}.
    std::map<std::string, std::vector<std::string>> basis;
    sktb::SkParameters hamiltonian;
    /// Absent means an orthogonal basis.
    std::optional<sktb::SkParameters> overlap;
    SpinSpec spin;
};

/// Shared implementation for the two parameterized Slater-Koster families.
///
/// SKTB and DFTB differ only in where the radial data comes from and whether
/// an overlap is mandatory; once resolved into an SkTable they are built
/// identically.
class SkTableModelBase : public Model {
public:
    const BasisSpec& basis() const override { return basis_; }

    const SpinSpec& spin() const override { return spin_; }

    /// H (and S, if parameterized) from the SK tables; v and r generic.
    ///
    /// Orthogonality is the hinge. Without an overlap the basis is orthogonal,
    /// so the commutator velocity and the diagonal position are exact and come
    /// for free. With one, neither is available: a non-orthogonal
    /// Slater-Koster model would have to differentiate its own tables, which
    /// is not implemented, and the generic route would be wrong.
    OperatorSources operator_sources() const override {
        const bool orthogonal = !table_.has_overlap;
        return OperatorSources{
            /*hamiltonian=*/OperatorSource::kModel,
            /*overlap=*/table_.has_overlap ? OperatorSource::kModel : OperatorSource::kUnavailable,
            /*velocity=*/orthogonal ? OperatorSource::kGeneric : OperatorSource::kUnavailable,
            /*position=*/orthogonal ? OperatorSource::kGeneric : OperatorSource::kUnavailable,
        };
    }

    std::unique_ptr<ImageContainer> BuildHamiltonianImages(AtomicData& atomic_data) const override {
        return construction_->build_hamiltonian_images(atomic_data);
    }

    std::unique_ptr<ImageContainer> BuildOverlapImages(AtomicData& atomic_data) const override {
        if (!table_.has_overlap) {
            return nullptr;
        }
        return construction_->build_overlap_images(atomic_data);
    }

    /// Returns false: an orthogonal Slater-Koster basis uses the generic
    /// image-based velocity and diagonal position construction.
    /// Returns the resolved parameter table.
    const sktb::SkTable& table() const { return table_; }

protected:
    SkTableModelBase(sktb::SkTable table, SpinSpec spin)
        : table_(std::move(table)), spin_(spin) {
        construction_ = std::make_shared<sktb::SkTableConstruction>(table_);
        basis_ = sktb::DeriveBasisSpec(table_, spin_);
    }

    sktb::SkTable table_;
    SpinSpec spin_;
    BasisSpec basis_;
    std::shared_ptr<sktb::SkTableConstruction> construction_;
};

class SktbModel : public SkTableModelBase {
public:
    /// Resolves SKTB parameters into a ready model.
    ///
    /// Args:
    ///   config: Basis, Hamiltonian parameters, and optional overlap.
    /// Returns:
    ///   A model whose basis() reports the orbital counts implied by the basis
    ///   labels and the cutoff radii implied by the radial functions.
    static std::unique_ptr<SktbModel> Create(const SktbConfig& config) {
        auto table = sktb::BuildSkTable(config.basis, config.hamiltonian, config.overlap);
        auto model = std::unique_ptr<SktbModel>(new SktbModel(std::move(table), config.spin));
        model->basis_.Validate("SKTB");
        return model;
    }

    std::string name() const override { return "SKTB"; }

private:
    SktbModel(sktb::SkTable table, SpinSpec spin)
        : SkTableModelBase(std::move(table), spin) {}
};

struct DftbConfig {
    std::map<std::string, std::vector<std::string>> basis;
    sktb::SkParameters hamiltonian;
    /// Required: a DFTB basis is non-orthogonal.
    sktb::SkParameters overlap;
    SpinSpec spin;
};

/// DFTB parameters read from a directory of Slater-Koster files.
struct DftbSkfConfig {
    std::string skf_dir;
    /// species symbol -> basis labels, e.g. {"C": {"s", "p"}}.
    ///
    /// Declared, not inferred. A .skf tabulates all ten channels up to d
    /// regardless of which shells the parameterization actually uses, and
    /// DFTB+ likewise takes the basis from its own input rather than the file.
    std::map<std::string, std::vector<std::string>> basis;
    sktb::SkfHeteronuclearPolicy heteronuclear = sktb::SkfHeteronuclearPolicy::kRejectAmbiguous;
    SpinSpec spin;
};

class DftbModel : public SkTableModelBase {
public:
    /// Resolves DFTB parameters into a ready model.
    ///
    /// Unlike SKTB, the overlap is mandatory: DFTB Slater-Koster data always
    /// carries both H and S tables.
    static std::unique_ptr<DftbModel> Create(const DftbConfig& config) {
        if (!config.overlap.IsEnabled()) {
            throw std::runtime_error(
                "DFTB setup requires overlap parameters: a DFTB basis is non-orthogonal. "
                "Use an SKTB model for an orthogonal Slater-Koster calculation.");
        }
        auto table = sktb::BuildSkTable(
            config.basis, config.hamiltonian, std::optional<sktb::SkParameters>(config.overlap));
        auto model = std::unique_ptr<DftbModel>(new DftbModel(std::move(table), config.spin));
        model->basis_.Validate("DFTB");
        return model;
    }

    /// Builds a DFTB model from a directory of .skf files.
    ///
    /// Reads one file per ordered species pair, taking onsite energies from
    /// the homonuclear files and both the H and S channels from every file.
    /// Everything after this point is the same machinery SKTB uses.
    static std::unique_ptr<DftbModel> FromSkfDirectory(const DftbSkfConfig& config) {
        DftbConfig resolved;
        resolved.basis = config.basis;
        resolved.spin = config.spin;
        sktb::ReadSkfParameterSet(
            config.skf_dir, config.basis, config.heteronuclear,
            &resolved.hamiltonian, &resolved.overlap);
        return Create(resolved);
    }

    std::string name() const override { return "DFTB"; }

private:
    DftbModel(sktb::SkTable table, SpinSpec spin)
        : SkTableModelBase(std::move(table), spin) {}
};

}  // namespace model
}  // namespace rsatb

#endif  // RSATB_MODEL_SKTB_SKTB_MODEL_HPP

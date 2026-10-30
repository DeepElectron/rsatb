#ifndef RSATB_OPERATOR_IMAGES_HPP
#define RSATB_OPERATOR_IMAGES_HPP

// H(R): real, or complex when spin-orbit coupling makes it so.
//
// A spin-degenerate Hamiltonian is real in real space, and storing it real
// halves both its memory and the cost of every Bloch sum over it. Spin-orbit
// coupling makes H(R) genuinely complex. Which one a model has follows from
// its spin treatment, so it is carried by the type -- the model builds one or
// the other, and every consumer is templated on the scalar and reached through
// Visit -- rather than detected from the values. S(R) and the position
// integrals stay real either way: the spinor overlap is S (x) I2.

#include "vbcsr/core/atomic/image_container.hpp"
#include "vbcsr/core/block_csr.hpp"

#include <complex>
#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace rsatb {
namespace op {

using RealImages = vbcsr::atomic::ImageContainer<double>;
using ComplexImages = vbcsr::atomic::ImageContainer<std::complex<double>>;

class HamiltonianImages {
public:
    HamiltonianImages() = default;
    HamiltonianImages(std::unique_ptr<RealImages> images) : images_(std::move(images)) {}
    HamiltonianImages(std::unique_ptr<ComplexImages> images) : images_(std::move(images)) {}

    /// Whether any images are held.
    explicit operator bool() const {
        return std::visit([](const auto& images) { return images != nullptr; }, images_);
    }

    bool is_complex() const { return std::holds_alternative<std::unique_ptr<ComplexImages>>(images_); }

    /// Calls `f` with the held container, as `ImageContainer<T>&`.
    template <typename F>
    decltype(auto) Visit(F&& f) const {
        return std::visit(
            [&](const auto& images) -> decltype(auto) {
                if (!images) throw std::runtime_error("No Hamiltonian images are held.");
                return f(*images);
            },
            images_);
    }

    /// H(k): complex whatever the storage.
    vbcsr::BlockSpMat<std::complex<double>>* sample_k(
        const std::vector<double>& kpoint, vbcsr::atomic::PhaseConvention convention) const {
        return Visit([&](auto& images) { return images.sample_k(kpoint, convention); });
    }

    /// The real container, for code that exists only for spin-degenerate H.
    ///
    /// Throws:
    ///   std::runtime_error if H(R) is complex.
    RealImages& real() const {
        if (is_complex()) {
            throw std::runtime_error(
                "This path needs a real H(R), but the Hamiltonian is complex (spin-orbit).");
        }
        const auto& images = std::get<std::unique_ptr<RealImages>>(images_);
        if (!images) throw std::runtime_error("No Hamiltonian images are held.");
        return *images;
    }

private:
    std::variant<std::unique_ptr<RealImages>, std::unique_ptr<ComplexImages>> images_;
};

}  // namespace op
}  // namespace rsatb

#endif  // RSATB_OPERATOR_IMAGES_HPP

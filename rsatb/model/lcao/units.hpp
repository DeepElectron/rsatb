#ifndef RSATB_MODEL_LCAO_UNITS_HPP
#define RSATB_MODEL_LCAO_UNITS_HPP

// op2c's unit convention, at the one boundary that crosses it.
//
// op2c works in Bohr; AtomicData positions, graph cutoffs and the real-space
// grids work in Angstrom. Every quantity that crosses is converted where it
// crosses, and both sides of that boundary -- the integral engine and the
// real-space orbital evaluation -- name the factor from here rather than
// spelling it out.

namespace rsatb {
namespace model {
namespace lcao {

constexpr double kBohrPerAngstrom = 1.8897259886;
constexpr double kAngstromPerBohr = 1.0 / kBohrPerAngstrom;

}  // namespace lcao
}  // namespace model
}  // namespace rsatb

#endif  // RSATB_MODEL_LCAO_UNITS_HPP

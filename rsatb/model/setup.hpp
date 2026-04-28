#ifndef RSATB_MODEL_SETUP_HPP
#define RSATB_MODEL_SETUP_HPP

#include "sktb/sktb.hpp"

#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace rsatb {
namespace model {

struct PreconditionerOptions {
    std::string mode = "H/S";
    double tol = 1e-7;
    double sptol = 1e-5;
    int maxiter = 20;
    std::string method = "graph";

    std::string normalized_mode() const {
        if (mode == "H/S") {
            return "S";
        }
        return mode;
    }
};

struct SetupOptions {
    std::vector<std::vector<double>> kpoints = {{0.0, 0.0, 0.0}};
    int nsamples = 1;
    bool compute_bounds = true;
    std::optional<std::pair<double, double>> energy_bounds;
    double bounds_zero_tol = 1e-12;
    int bounds_max_matvecs = 10000;
    PreconditionerOptions preconditioner;
};

struct Hdf5Input {
    std::string filename;
    bool lspinorb = false;
    int nspin = 1;
    bool noncolin = false;

    bool enabled() const {
        return !filename.empty();
    }
};

struct LCAOIntegralInput {
    std::vector<std::string> orb_names;
    std::string orb_dir;
    std::vector<std::string> psd_names;
    std::string psd_dir;
    int nspin = 1;
    bool lspinorb = false;
    bool noncolin = false;
    std::string log_file = "op2c.log";

    bool has_overlap_inputs() const {
        return !orb_names.empty() && !orb_dir.empty();
    }

    bool has_velocity_inputs() const {
        return has_overlap_inputs() && !psd_names.empty() && !psd_dir.empty();
    }

    bool enabled() const {
        return has_overlap_inputs();
    }
};

using SKTBNativeInput = rsatb::backend::SKTBModel;
using DFTBNativeInput = rsatb::backend::SKTBModel;

struct LCAOModelConfig {
    std::optional<Hdf5Input> input;
    LCAOIntegralInput integrals;
};

struct SKTBModelConfig {
    std::optional<Hdf5Input> input;
    SKTBNativeInput native;
};

struct DFTBModelConfig {
    std::optional<Hdf5Input> input;
    DFTBNativeInput native;
};

using AnyModelConfig = std::variant<LCAOModelConfig, SKTBModelConfig, DFTBModelConfig>;

} // namespace model
} // namespace rsatb

#endif // RSATB_MODEL_SETUP_HPP

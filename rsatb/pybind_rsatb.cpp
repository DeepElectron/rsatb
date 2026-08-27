#include "utils/interrupt.hpp"
#include "utils/profile.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include "rsatb.hpp"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace py = pybind11;


PYBIND11_MODULE(_rsatb, m) {
    m.doc() = "RSATB Python Binding";

    // --- io layer ---------------------------------------------------------

    py::class_<vbcsr::atomic::Structure>(m, "Structure",
        "Atomic geometry: positions, species, cell, periodicity.\n\n"
        "Carries no orbital or cutoff information -- that comes from the model.")
        .def(py::init<>())
        .def_static("from_file",
            [](const std::string& filename, const std::string& format,
               const std::array<int, 3>& supercell) {
                auto structure = vbcsr::atomic::ReadStructure(filename, format);
                structure.supercell = supercell;
                return structure;
            },
            py::arg("filename"), py::arg("format") = "",
            py::arg("supercell") = std::array<int, 3>{{1, 1, 1}})
        .def_readwrite("pos", &vbcsr::atomic::Structure::pos)
        .def_readwrite("z", &vbcsr::atomic::Structure::z)
        .def_readwrite("cell", &vbcsr::atomic::Structure::cell)
        .def_readwrite("pbc", &vbcsr::atomic::Structure::pbc)
        .def_readwrite("n_atom", &vbcsr::atomic::Structure::n_atom)
        .def_readwrite("supercell", &vbcsr::atomic::Structure::supercell)
        .def("type_symbols", &vbcsr::atomic::Structure::TypeSymbols)
        .def("expanded", &vbcsr::atomic::Structure::BuildSupercell,
             "Return the geometry after applying the structure's supercell factors.")
        .def_static(
            "from_arrays",
            [](std::vector<double> pos, std::vector<int> z, std::vector<double> cell,
               std::vector<bool> pbc, const std::array<int, 3>& supercell) {
                vbcsr::atomic::Structure structure;
                if (pos.size() != z.size() * 3) {
                    throw std::runtime_error(
                        "Structure.from_arrays: pos must hold three values per atom (" +
                        std::to_string(z.size()) + " atoms, " + std::to_string(pos.size()) +
                        " coordinates).");
                }
                if (cell.size() != 9) {
                    throw std::runtime_error(
                        "Structure.from_arrays: cell must hold nine values, row-major.");
                }
                if (pbc.size() != 3) {
                    throw std::runtime_error("Structure.from_arrays: pbc must hold three flags.");
                }
                structure.n_atom = static_cast<int>(z.size());
                structure.pos = std::move(pos);
                structure.z = std::move(z);
                structure.cell = std::move(cell);
                structure.pbc = std::move(pbc);
                structure.supercell = supercell;
                return structure;
            },
            py::arg("pos"), py::arg("z"), py::arg("cell"),
            py::arg("pbc") = std::vector<bool>{true, true, true},
            py::arg("supercell") = std::array<int, 3>{{1, 1, 1}},
            "Build a Structure from arrays, with the shapes checked.")
        .def("__repr__", [](const vbcsr::atomic::Structure& self) {
            std::string species;
            for (const auto& symbol : self.TypeSymbols()) {
                if (!species.empty()) species += ", ";
                species += symbol;
            }
            return "<Structure n_atom=" + std::to_string(self.n_atom) +
                   " species=[" + species + "] supercell=[" +
                   std::to_string(self.supercell[0]) + ", " +
                   std::to_string(self.supercell[1]) + ", " +
                   std::to_string(self.supercell[2]) + "]>";
        });

    // Ctrl-C arrives as InterruptedError from a step boundary inside the
    // C++ loops (utils/interrupt.hpp -- CPython's own deferred handling never
    // fires during an hours-long extension call). Mapping it onto
    // KeyboardInterrupt gives the Python side the exception Ctrl-C ordinarily
    // means, so the CLI and scripts handle it with the ordinary idiom.
    pybind11::register_exception<rsatb::util::InterruptedError>(
        m, "Interrupted", PyExc_KeyboardInterrupt);

    // Verbosity is process-wide state rather than a per-object setting: the
    // solver logs from many call sites that never see the calculation object,
    // so threading a level through every one of them would be the `verbose`
    // bool's problem again. Set it once before building a calculation --
    // RSATB deliberately takes no verbosity argument, because when it did,
    // the two spellings could disagree and the bool silently won.
    m.def("set_log_level", [](const std::string& level) {
        using rsatb::util::Log;
        using rsatb::util::LogLevel;
        if (level == "silent") Log::level() = LogLevel::kSilent;
        else if (level == "summary") Log::level() = LogLevel::kSummary;
        else if (level == "progress") Log::level() = LogLevel::kProgress;
        else if (level == "debug") Log::level() = LogLevel::kDebug;
        else throw std::invalid_argument(
            "unknown log level '" + level + "'; expected silent, summary, progress or debug");
    }, pybind11::arg("level"),
       "Set solver verbosity: silent, summary, progress or debug.");

    // Setup prints its own table when it finishes, because a setup that goes
    // wrong is usually abandoned before any task runs and the numbers are still
    // wanted. This prints the whole run: the same setup phases plus whatever the
    // tasks added, against wall time since setup began. The registry is never
    // cleared between the two, so calling this is the only way to see task rows.
    m.def("profile_report", []() {
        // Over the world, matching what setup reports against: a table that
        // silently described rank 0 alone would be the thing the reduce exists
        // to prevent. Every rank must reach this, so it is called from
        // run_tasks() on all of them, not just the writing rank.
        int initialized = 0;
        MPI_Initialized(&initialized);
        rsatb::util::Profile::report(0.0, initialized ? MPI_COMM_WORLD : MPI_COMM_NULL);
    }, "Print the accumulated phase profile for the whole run.");

    m.def("atomic_number", [](const std::string& symbol) {
              return vbcsr::atomic::AtomicNumber(vbcsr::atomic::CanonicalSymbol(symbol));
          },
          py::arg("symbol"),
          "Atomic number of a chemical symbol, e.g. 'C' -> 6. A decimal string is\n"
          "accepted too, so '6' also resolves to carbon. Raises on anything else.");

    py::class_<rsatb::io::HamiltonianFile>(m, "HamiltonianFile",
        "A precomputed real-space Hamiltonian file and the layout it uses.")
        .def(py::init([](std::string path, const std::string& format) {
                 rsatb::io::HamiltonianFile file;
                 file.path = std::move(path);
                 file.format = rsatb::io::ParseHamiltonianFormat(format);
                 return file;
             }),
             py::arg("path"), py::arg("format") = "deeptb")
        .def_readwrite("path", &rsatb::io::HamiltonianFile::path)
        .def_property(
            "format",
            [](const rsatb::io::HamiltonianFile& self) { return rsatb::io::ToString(self.format); },
            [](rsatb::io::HamiltonianFile& self, const std::string& value) {
                self.format = rsatb::io::ParseHamiltonianFormat(value);
            })
        .def("__repr__", [](const rsatb::io::HamiltonianFile& self) {
            return "<HamiltonianFile path='" + self.path + "' format='" +
                   rsatb::io::ToString(self.format) + "'>";
        });

    // --- model layer ------------------------------------------------------

    py::class_<rsatb::model::SpinSpec>(m, "SpinSpec",
        "How spin enters the basis: none, collinear, noncollinear, or spinorbit.")
        .def(py::init([](const std::string& treatment) {
                 return rsatb::model::ParseSpinSpec(treatment);
             }),
             py::arg("treatment") = "none")
        .def_property(
            "treatment",
            [](const rsatb::model::SpinSpec& self) { return rsatb::model::ToString(self.treatment); },
            [](rsatb::model::SpinSpec& self, const std::string& value) {
                self = rsatb::model::ParseSpinSpec(value);
            })
        .def("components", &rsatb::model::SpinSpec::Components)
        .def("channels", &rsatb::model::SpinSpec::Channels)
        .def("__repr__", [](const rsatb::model::SpinSpec& self) {
            return "<SpinSpec '" + rsatb::model::ToString(self.treatment) + "' components=" +
                   std::to_string(self.Components()) + ">";
        });

    py::class_<rsatb::model::BasisSpec>(m, "BasisSpec",
        "The basis a model works in: species, orbitals per species, and graph\n"
        "cutoff radii. Derived by the model -- never supplied by the caller.")
        .def_readonly("type_symbols", &rsatb::model::BasisSpec::type_symbols)
        .def_readonly("type_norb", &rsatb::model::BasisSpec::type_norb)
        .def_readonly("cutoff_radius", &rsatb::model::BasisSpec::cutoff_radius)
        .def("__repr__", [](const rsatb::model::BasisSpec& self) {
            std::string body;
            for (size_t i = 0; i < self.type_symbols.size(); ++i) {
                if (!body.empty()) body += ", ";
                std::ostringstream cutoff;
                cutoff.precision(4);
                cutoff << std::fixed << self.cutoff_radius[i];
                body += self.type_symbols[i] + ": " + std::to_string(self.type_norb[i]) +
                        " orb, r<=" + cutoff.str() + " A";
            }
            return "<BasisSpec " + body + ">";
        });

    py::class_<rsatb::model::Capabilities>(m, "Capabilities",
        "Which operators are available once a model is set up.")
        .def_readonly("overlap", &rsatb::model::Capabilities::overlap)
        .def_readonly("velocity", &rsatb::model::Capabilities::velocity)
        .def_readonly("position", &rsatb::model::Capabilities::position)
        .def("orthogonal", &rsatb::model::Capabilities::orthogonal)
        .def("__repr__", [](const rsatb::model::Capabilities& self) {
            std::string body = self.overlap ? "non-orthogonal" : "orthogonal";
            if (self.velocity) body += ", velocity";
            if (self.position) body += ", position";
            return "<Capabilities " + body + ">";
        });

    py::class_<rsatb::model::OrbitalFiles>(m, "OrbitalFiles",
        "Numerical atomic orbital files, one per species in ascending-Z order.")
        .def(py::init([](std::vector<std::string> names, std::string dir) {
                 return rsatb::model::OrbitalFiles{std::move(names), std::move(dir)};
             }),
             py::arg("names"), py::arg("dir") = "")
        .def_readwrite("names", &rsatb::model::OrbitalFiles::names)
        .def_readwrite("dir", &rsatb::model::OrbitalFiles::dir);

    py::class_<rsatb::model::PseudopotentialFiles>(m, "PseudopotentialFiles",
        "Pseudopotential files, in the same order as the orbital files.\n\n"
        "Optional: kept for the explicit-integral cross-check velocity; the "
        "production commutator velocity needs no pseudopotentials.")
        .def(py::init([](std::vector<std::string> names, std::string dir) {
                 return rsatb::model::PseudopotentialFiles{std::move(names), std::move(dir)};
             }),
             py::arg("names") = std::vector<std::string>{}, py::arg("dir") = "")
        .def_readwrite("names", &rsatb::model::PseudopotentialFiles::names)
        .def_readwrite("dir", &rsatb::model::PseudopotentialFiles::dir);

    py::class_<rsatb::model::PreconditionerOptions>(m, "PreconditionerOptions",
        "Preconditioner settings for the iterative solvers.")
        .def(py::init([](const std::string& operator_form,
                         double tol, double sptol, int maxiter) {
                 rsatb::model::PreconditionerOptions options;
                 options.operator_form = rsatb::model::ParseOperatorForm(operator_form);
                 options.tol = tol;
                 options.sptol = sptol;
                 options.maxiter = maxiter;
                 return options;
             }),
             py::arg("operator_form") = "solve",
             py::arg("tol") = 1e-7,
             py::arg("sptol") = 1e-5,
             py::arg("maxiter") = 20)
        .def_property(
            "operator_form",
            [](const rsatb::model::PreconditionerOptions& self) {
                return self.operator_form == rsatb::model::OperatorForm::kSandwich
                           ? "sandwich" : "solve";
            },
            [](rsatb::model::PreconditionerOptions& self, const std::string& value) {
                self.operator_form = rsatb::model::ParseOperatorForm(value);
            })
        .def_readwrite("tol", &rsatb::model::PreconditionerOptions::tol)
        .def_readwrite("sptol", &rsatb::model::PreconditionerOptions::sptol)
        .def_readwrite("maxiter", &rsatb::model::PreconditionerOptions::maxiter)
        .def("__repr__", [](const rsatb::model::PreconditionerOptions& self) {
            const std::string form =
                self.operator_form == rsatb::model::OperatorForm::kSandwich
                    ? "sandwich" : "solve";
            return "<PreconditionerOptions operator_form='" + form + "'>";
        });

    py::class_<rsatb::model::SetupOptions>(m, "SetupOptions",
        "What the calculation does: k-points, sampling, spectral bounds, and\n"
        "preconditioning. Independent of which model family supplies operators.")
        .def(py::init([](std::vector<std::vector<double>> kpoints,
                         int nsamples,
                         unsigned long sample_seed,
                         bool compute_bounds,
                         py::object energy_bounds,
                         double bounds_zero_tol,
                         int bounds_max_matvecs,
                         rsatb::model::PreconditionerOptions preconditioner) {
                 rsatb::model::SetupOptions options;
                 options.kpoints = std::move(kpoints);
                 options.nsamples = nsamples;
                 options.sample_seed = sample_seed;
                 options.compute_bounds = compute_bounds;
                 if (!energy_bounds.is_none()) {
                     const auto bounds = py::cast<std::vector<double>>(energy_bounds);
                     if (bounds.size() != 2) {
                         throw std::runtime_error(
                             "energy_bounds must contain exactly two values: [emin, emax].");
                     }
                     options.energy_bounds = std::make_pair(bounds[0], bounds[1]);
                 }
                 options.bounds_zero_tol = bounds_zero_tol;
                 options.bounds_max_matvecs = bounds_max_matvecs;
                 options.preconditioner = std::move(preconditioner);
                 return options;
             }),
             py::arg("kpoints") = std::vector<std::vector<double>>{{0.0, 0.0, 0.0}},
             py::arg("nsamples") = 1,
             py::arg("sample_seed") = 20240917UL,
             py::arg("compute_bounds") = true,
             py::arg("energy_bounds") = py::none(),
             py::arg("bounds_zero_tol") = 1e-12,
             py::arg("bounds_max_matvecs") = 10000,
             py::arg("preconditioner") = rsatb::model::PreconditionerOptions{})
        .def_readwrite("kpoints", &rsatb::model::SetupOptions::kpoints)
        .def_readwrite("nsamples", &rsatb::model::SetupOptions::nsamples)
        .def_readwrite("sample_seed", &rsatb::model::SetupOptions::sample_seed)
        .def_readwrite("compute_bounds", &rsatb::model::SetupOptions::compute_bounds)
        .def_readwrite("bounds_zero_tol", &rsatb::model::SetupOptions::bounds_zero_tol)
        .def_readwrite("bounds_max_matvecs", &rsatb::model::SetupOptions::bounds_max_matvecs)
        .def_readwrite("preconditioner", &rsatb::model::SetupOptions::preconditioner)
        // Constructor-only until now: you could pass energy_bounds but not read
        // it back or change it, which is exactly the asymmetry this redesign
        // set out to remove.
        .def_property(
            "energy_bounds",
            [](const rsatb::model::SetupOptions& self) -> py::object {
                if (!self.energy_bounds.has_value()) return py::none();
                return py::make_tuple(self.energy_bounds->first, self.energy_bounds->second);
            },
            [](rsatb::model::SetupOptions& self, py::object value) {
                if (value.is_none()) {
                    self.energy_bounds.reset();
                    return;
                }
                const auto bounds = py::cast<std::vector<double>>(value);
                if (bounds.size() != 2) {
                    throw std::runtime_error(
                        "energy_bounds must contain exactly two values: [emin, emax].");
                }
                self.energy_bounds = std::make_pair(bounds[0], bounds[1]);
            })
        .def("__repr__", [](const rsatb::model::SetupOptions& self) {
            std::string bounds = "auto";
            if (self.energy_bounds.has_value()) {
                std::ostringstream out;
                out.precision(3);
                out << "[" << self.energy_bounds->first << ", " << self.energy_bounds->second << "]";
                bounds = out.str();
            } else if (!self.compute_bounds) {
                bounds = "none";
            }
            return "<SetupOptions nk=" + std::to_string(self.kpoints.size()) +
                   " nsamples=" + std::to_string(self.nsamples) +
                   " bounds=" + bounds + ">";
        });

    py::class_<rsatb::model::Model>(m, "Model",
        "Base class of every model family. Reports the basis it works in and\n"
        "which operators it can build.")
        .def_property_readonly("name", &rsatb::model::Model::name)
        .def_property_readonly("basis", &rsatb::model::Model::basis,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("spin", &rsatb::model::Model::spin,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("capabilities", &rsatb::model::Model::capabilities)
        .def("__repr__", [](const rsatb::model::Model& self) {
            const auto& basis = self.basis();
            std::string species;
            for (size_t i = 0; i < basis.type_symbols.size(); ++i) {
                if (!species.empty()) species += ", ";
                species += basis.type_symbols[i] + "(" + std::to_string(basis.type_norb[i]) + ")";
            }
            return "<" + self.name() + " " + species + " " +
                   std::string(self.capabilities().orthogonal() ? "orthogonal" : "non-orthogonal") + ">";
        });

    // --- Slater-Koster parameters -----------------------------------------

    py::class_<rsatb::model::sktb::SkRadialFunction>(m, "SkRadialFunction",
        "Distance dependence of one Slater-Koster channel.\n\n"
        "Build with sk_spline(...) or sk_exp_poly(...).")
        .def_readwrite("kind", &rsatb::model::sktb::SkRadialFunction::kind)
        .def_readwrite("r", &rsatb::model::sktb::SkRadialFunction::r)
        .def_readwrite("values", &rsatb::model::sktb::SkRadialFunction::values)
        .def_readwrite("coeffs", &rsatb::model::sktb::SkRadialFunction::coeffs)
        .def_readwrite("lambda_value", &rsatb::model::sktb::SkRadialFunction::lambda_value)
        .def_readwrite("r_on", &rsatb::model::sktb::SkRadialFunction::r_on)
        .def_readwrite("r_cut", &rsatb::model::sktb::SkRadialFunction::r_cut)
        .def("cutoff", &rsatb::model::sktb::SkRadialFunction::Cutoff)
        .def("evaluate", &rsatb::model::sktb::SkRadialFunction::Evaluate, py::arg("distance"));

    m.def("sk_spline",
          [](std::vector<double> r, std::vector<double> values, py::object r_cut) {
              const double cutoff = r_cut.is_none() ? 0.0 : py::cast<double>(r_cut);
              return rsatb::model::sktb::MakeSpline(std::move(r), std::move(values), cutoff);
          },
          py::arg("r"), py::arg("values"), py::arg("r_cut") = py::none(),
          "Cubic-spline radial function through tabulated values.");

    m.def("sk_exp_poly",
          [](std::vector<double> coeffs, double r_cut, double lambda_value, double r_on) {
              return rsatb::model::sktb::MakeExpPoly(std::move(coeffs), r_cut, lambda_value, r_on);
          },
          py::arg("coeffs"), py::arg("r_cut"), py::arg("lambda_value") = 0.0, py::arg("r_on") = 0.0,
          "Exponential-times-polynomial radial function with a cosine taper.");

    py::class_<rsatb::model::sktb::SkParameters>(m, "SkParameters",
        "Onsite energies and pair channels for ONE matrix, H or S.\n\n"
        "H and S are separate objects, so there is nothing to disambiguate.")
        .def(py::init([](py::dict onsite, py::dict pairs) {
                 rsatb::model::sktb::SkParameters parameters;
                 for (const auto& species_entry : onsite) {
                     const auto species = py::cast<std::string>(species_entry.first);
                     for (const auto& orbital_entry : py::cast<py::dict>(species_entry.second)) {
                         parameters.SetOnsite(species,
                                              py::cast<std::string>(orbital_entry.first),
                                              py::cast<double>(orbital_entry.second));
                     }
                 }
                 // pairs: {("C", "C"): {"pp_pi": SkRadialFunction, ...}}
                 for (const auto& pair_entry : pairs) {
                     const auto species = py::cast<std::vector<std::string>>(pair_entry.first);
                     if (species.size() != 2) {
                         throw std::runtime_error(
                             "An SkParameters pair key must name exactly two species.");
                     }
                     auto& pair = parameters.Pair(species[0], species[1]);
                     for (const auto& channel_entry : py::cast<py::dict>(pair_entry.second)) {
                         pair.channels[py::cast<std::string>(channel_entry.first)] =
                             py::cast<rsatb::model::sktb::SkRadialFunction>(channel_entry.second);
                     }
                 }
                 return parameters;
             }),
             py::arg("onsite") = py::dict(),
             py::arg("pairs") = py::dict(),
             "Onsite energies and pair channels for one matrix (H or S).\n\n"
             "onsite: {species: {orbital: value}}\n"
             "pairs:  {(row_species, col_species): {channel_name: SkRadialFunction}}\n"
             "Channel names use the shell-pair plus bond-type form, e.g. \"pp_pi\".");

    py::class_<rsatb::model::SktbModel, rsatb::model::Model>(m, "SktbModel",
        "Slater-Koster model from parameterized radial functions.\n\n"
        "Orthogonal unless overlap parameters are given.")
        .def(py::init([](std::map<std::string, std::vector<std::string>> basis,
                         rsatb::model::sktb::SkParameters hamiltonian,
                         py::object overlap,
                         const std::string& spin) {
                 rsatb::model::SktbConfig config;
                 config.basis = std::move(basis);
                 config.hamiltonian = std::move(hamiltonian);
                 if (!overlap.is_none()) {
                     config.overlap = py::cast<rsatb::model::sktb::SkParameters>(overlap);
                 }
                 config.spin = rsatb::model::ParseSpinSpec(spin);
                 return rsatb::model::SktbModel::Create(config);
             }),
             py::arg("basis"),
             py::arg("hamiltonian"),
             py::arg("overlap") = py::none(),
             py::arg("spin") = "none");

    py::class_<rsatb::model::DftbModel, rsatb::model::Model>(m, "DftbModel",
        "DFTB model: Slater-Koster tables, non-orthogonal by construction.\n\n"
        "Use DftbModel.from_skf_directory(...) to read a .skf parameter set.")
        .def(py::init([](std::map<std::string, std::vector<std::string>> basis,
                         rsatb::model::sktb::SkParameters hamiltonian,
                         rsatb::model::sktb::SkParameters overlap,
                         const std::string& spin) {
                 rsatb::model::DftbConfig config;
                 config.basis = std::move(basis);
                 config.hamiltonian = std::move(hamiltonian);
                 config.overlap = std::move(overlap);
                 config.spin = rsatb::model::ParseSpinSpec(spin);
                 return rsatb::model::DftbModel::Create(config);
             }),
             py::arg("basis"),
             py::arg("hamiltonian"),
             py::arg("overlap"),
             py::arg("spin") = "none")
        .def_static(
            "from_skf_directory",
            [](const std::string& skf_dir,
               std::map<std::string, std::vector<std::string>> basis,
               const std::string& heteronuclear,
               const std::string& spin) {
                rsatb::model::DftbSkfConfig config;
                config.skf_dir = skf_dir;
                config.basis = std::move(basis);
                const std::string policy = vbcsr::atomic::LowerCopy(vbcsr::atomic::TrimCopy(heteronuclear));
                if (policy == "reject_ambiguous") {
                    config.heteronuclear = rsatb::model::sktb::SkfHeteronuclearPolicy::kRejectAmbiguous;
                } else if (policy == "assume_lower_shell_on_row_species") {
                    config.heteronuclear =
                        rsatb::model::sktb::SkfHeteronuclearPolicy::kAssumeLowerShellOnRowSpecies;
                } else {
                    throw std::runtime_error(
                        "heteronuclear must be 'reject_ambiguous' or 'assume_lower_shell_on_row_species'.");
                }
                config.spin = rsatb::model::ParseSpinSpec(spin);
                return rsatb::model::DftbModel::FromSkfDirectory(config);
            },
            py::arg("skf_dir"),
            py::arg("basis"),
            py::arg("heteronuclear") = "reject_ambiguous",
            py::arg("spin") = "none",
            "Build a DFTB model from a directory of Slater-Koster (.skf) files.\n\n"
            "The basis is declared, not inferred: a .skf tabulates all ten channels up to d\n"
            "regardless of which shells the parameterization uses.\n\n"
            "Heteronuclear pairs work whenever the declared bases force a unique reading of\n"
            "each mixed-shell channel, which is the usual case. A pair where both species carry\n"
            "both shells of a channel is genuinely ambiguous in the file format and is refused.");

    py::class_<rsatb::model::TightBindingFileModel, rsatb::model::Model>(m, "TightBindingFileModel",
        "Orthogonal tight-binding model read from a Hamiltonian file.\n\n"
        "The one model that must be told its basis: the supported file formats\n"
        "carry no basis header.")
        .def(py::init([](rsatb::io::HamiltonianFile hamiltonian,
                         std::vector<std::string> type_symbols,
                         std::vector<int> type_norb,
                         std::vector<double> cutoff_radius,
                         const std::string& spin) {
                 rsatb::model::TightBindingFileConfig config;
                 config.hamiltonian = std::move(hamiltonian);
                 config.type_symbols = std::move(type_symbols);
                 config.type_norb = std::move(type_norb);
                 config.cutoff_radius = std::move(cutoff_radius);
                 config.spin = rsatb::model::ParseSpinSpec(spin);
                 return rsatb::model::TightBindingFileModel::Create(std::move(config), MPI_COMM_WORLD);
             }),
             py::arg("hamiltonian"),
             py::arg("type_symbols"),
             py::arg("type_norb"),
             py::arg("cutoff_radius"),
             py::arg("spin") = "none",
             "Orthogonal tight-binding model read from a precomputed Hamiltonian file.\n\n"
             "Unlike the other models the basis must be given: the supported file formats "
             "carry no basis header, so orbital counts cannot be derived.");

    py::class_<rsatb::model::LcaoModel, rsatb::model::Model>(m, "LcaoModel",
        "LCAO model: a precomputed Hamiltonian plus numerical atomic orbitals.\n\n"
        "Always non-orthogonal, so orbital files are required. The basis --\n"
        "orbital counts and cutoff radii -- is read from those files.")
        .def(py::init([](rsatb::io::HamiltonianFile hamiltonian,
                         rsatb::model::OrbitalFiles orbitals,
                         rsatb::model::PseudopotentialFiles pseudopotentials,
                         const std::string& spin,
                         std::string log_file) {
                 rsatb::model::LcaoConfig config;
                 config.hamiltonian = std::move(hamiltonian);
                 config.orbitals = std::move(orbitals);
                 config.pseudopotentials = std::move(pseudopotentials);
                 config.spin = rsatb::model::ParseSpinSpec(spin);
                 config.log_file = std::move(log_file);
                 return rsatb::model::LcaoModel::Create(std::move(config), MPI_COMM_WORLD);
             }),
             py::arg("hamiltonian"),
             py::arg("orbitals"),
             py::arg("pseudopotentials") = rsatb::model::PseudopotentialFiles{},
             py::arg("spin") = "none",
             py::arg("log_file") = "op2c.log");

    py::class_<rsatb::RSATB>(m, "RSATB")
        // The model is kept alive by the RSATB instance because velocity and
        // position operators are built lazily through it.
        .def(py::init([](const vbcsr::atomic::Structure& structure,
                         const rsatb::model::Model& model,
                         const rsatb::model::SetupOptions& options,
                         int npar) {
                 return new rsatb::RSATB(structure, model, options, npar);
             }),
             py::arg("structure"),
             py::arg("model"),
             py::arg("options") = rsatb::model::SetupOptions{},
             py::arg("npar") = 1,
             py::keep_alive<1, 3>())
        // Pre-partitioned graph plus a model. type_norb is not an argument:
        // it comes from the model's basis, so the block structure has one
        // source of truth.
        .def(py::init([](py::array_t<int> atom_index, py::array_t<int> atom_type,
                         py::array_t<int> edge_index, py::array_t<int> edge_shift_vec,
                         py::array_t<double> cell, py::array_t<double> pos,
                         const rsatb::model::Model& model,
                         const rsatb::model::SetupOptions& options,
                         size_t N_atom, size_t atom_offset, size_t N_edge,
                         int npar) {
                 rsatb::RSATB::AtomicGraphArrays graph;
                 graph.n_atom = static_cast<size_t>(atom_index.size());
                 graph.n_edge = static_cast<size_t>(edge_index.size() / 2);
                 graph.N_atom = (N_atom == 0) ? graph.n_atom : N_atom;
                 graph.N_edge = (N_edge == 0) ? graph.n_edge : N_edge;
                 graph.atom_offset = atom_offset;
                 graph.atom_index = atom_index.data();
                 graph.atom_type = atom_type.data();
                 graph.edge_index = edge_index.data();
                 graph.edge_shift_vec = edge_shift_vec.data();
                 graph.cell = cell.data();
                 graph.pos = pos.data();
                 return new rsatb::RSATB(graph, model, options, npar);
             }),
             py::arg("atom_index"), py::arg("atom_type"), py::arg("edge_index"),
             py::arg("edge_shift_vec"), py::arg("cell"), py::arg("pos"),
             py::arg("model"),
             py::arg("options") = rsatb::model::SetupOptions{},
             py::arg("N_atom") = 0, py::arg("atom_offset") = 0, py::arg("N_edge") = 0,
             py::arg("npar") = 1,
             py::keep_alive<1, 8>())
        .def("get_kernel_info", &rsatb::RSATB::get_kernel_info)
         .def("test_solve", &rsatb::RSATB::test_solve)
        .def("compute_dos", [](rsatb::RSATB& self, double dE, std::string method) {
             std::vector<double> dos_out;
             std::vector<double> omega_out;
             self.compute_dos(dE, dos_out, omega_out, method);
             return std::make_pair(dos_out, omega_out);
         }, py::arg("dE"), py::arg("method") = "kpm")
        .def("cell", &rsatb::RSATB::cell)
        .def("energy_bounds", &rsatb::RSATB::energy_bounds)
        .def("compute_rs_ldos", [](rsatb::RSATB& self, std::vector<double> energies, double eta,
                                   double spacing, int box_edge, int n_samples, int block,
                                   unsigned long seed) {
             rsatb::calc::RealSpaceLdosOptions opt;
             opt.energies = std::move(energies);
             opt.eta = eta;
             opt.spacing = spacing;
             opt.box_edge = box_edge;
             opt.n_samples = n_samples;
             opt.block = block;
             opt.seed = seed;
             const auto res = self.compute_rs_ldos(std::move(opt));

             // This rank's compact share: global point indices plus values.
             // No rank holds the whole map by design; the serial case simply
             // owns everything.
             const int64_t n_owned = res.domain.n_point();
             py::array_t<int64_t> index(n_owned);
             auto* index_ptr = index.mutable_data();
             for (int64_t p = 0; p < n_owned; ++p) {
                 index_ptr[p] = rsatb::calc::GlobalIndexOfOwnedPoint(res.grid, res.domain, p);
             }
             const size_t n_energy = res.ldos.size();
             py::array_t<double> values({static_cast<py::ssize_t>(n_energy),
                                         static_cast<py::ssize_t>(n_owned)});
             auto* values_ptr = values.mutable_data();
             for (size_t k = 0; k < n_energy; ++k) {
                 std::copy(res.ldos[k].begin(), res.ldos[k].end(),
                           values_ptr + k * static_cast<size_t>(n_owned));
             }
             py::dict out;
             out["shape"] = py::make_tuple(res.grid.shape[0], res.grid.shape[1],
                                           res.grid.shape[2]);
             out["n_point"] = res.grid.n_point();
             out["point_index"] = index;
             out["ldos"] = values;
             out["filter_degree"] = res.filter_degree;
             out["n_states"] = res.n_states;
             out["sampler_residual"] = res.sampler_residual;
             return out;
         }, py::arg("energies"), py::arg("eta"), py::arg("spacing") = 0.4,
            py::arg("box_edge") = 0, py::arg("n_samples") = 64, py::arg("block") = 16,
            py::arg("seed") = 12345)
        .def("compute_ac", [](rsatb::RSATB& self, int nt, int n_kpm, double temp, double mu, std::string axis, std::string method, int block) {
             std::vector<double> ac_out;
             std::vector<double> omega_out;
             self.compute_ac(nt, n_kpm, temp, mu, ac_out, omega_out, axis, method, block);
             return std::make_pair(ac_out, omega_out);
         }, py::arg("nt"), py::arg("n_kpm"), py::arg("temp"), py::arg("mu"), py::arg("axis") = "abc",
            py::arg("method") = "concurrent", py::arg("block") = 0)
        .def("compute_dc_spectrum", [](rsatb::RSATB& self, int n_moments, std::string formulation, std::string axis, int partition_sections, size_t max_workspace_mb, std::string kernel, std::string method) {
             std::vector<double> sigma_out;
             std::vector<double> energy_out;
             self.compute_dc_spectrum(n_moments, sigma_out, energy_out, formulation, axis, partition_sections, max_workspace_mb, kernel, method);
             return std::make_pair(sigma_out, energy_out);
         },
         py::arg("n_moments"),
         py::arg("formulation") = "greenwood",
         py::arg("axis") = "abc",
         py::arg("partition_sections") = 0,
         py::arg("max_workspace_mb") = 0,
         py::arg("kernel") = "jackson",
         py::arg("method") = "fftcheb")
        .def("compute_dc", [](rsatb::RSATB& self, int n_moments, double mu, double temp, std::string formulation, std::string axis, int partition_sections, size_t max_workspace_mb, std::string kernel, std::string method) {
             std::vector<double> dc_out;
             self.compute_dc(n_moments, mu, temp, dc_out, formulation, axis, partition_sections, max_workspace_mb, kernel, method);
             return dc_out;
         },
         py::arg("n_moments"),
         py::arg("mu"),
         py::arg("temp"),
         py::arg("formulation") = "greenwood",
         py::arg("axis") = "abc",
         py::arg("partition_sections") = 0,
         py::arg("max_workspace_mb") = 0,
         py::arg("kernel") = "jackson",
         py::arg("method") = "fftcheb")
        .def("compute_occupation", [](rsatb::RSATB& self, double mu, double temp, double indicator_threshold, int n_max) {
             const auto r = self.compute_occupation(mu, temp, indicator_threshold, n_max);
             py::dict d;
             d["n_electrons"] = r.n_electrons;
             d["band_energy"] = r.band_energy;
             d["mu"] = r.mu;
             d["n_moments"] = r.n_moments;
             d["indicator"] = r.indicator;
             d["indicator_fired"] = r.indicator_fired;
             return d;
         },
         py::arg("mu"),
         py::arg("temp"),
         py::arg("indicator_threshold") = 1e-4,
         py::arg("n_max") = 1 << 16)
        .def("find_fermi_level", [](rsatb::RSATB& self, double n_electrons, double temp, double indicator_threshold, int n_max, double mu_guess) {
             const auto r = self.find_fermi_level(n_electrons, temp, indicator_threshold, n_max, mu_guess);
             py::dict d;
             d["n_electrons"] = r.n_electrons;
             d["band_energy"] = r.band_energy;
             d["mu"] = r.mu;
             d["n_moments"] = r.n_moments;
             d["indicator"] = r.indicator;
             d["indicator_fired"] = r.indicator_fired;
             return d;
         },
         py::arg("n_electrons"),
         py::arg("temp"),
         py::arg("indicator_threshold") = 1e-4,
         py::arg("n_max") = 1 << 16,
         py::arg("mu_guess") = std::numeric_limits<double>::quiet_NaN())
        .def("compute_msd_spectrum", [](rsatb::RSATB& self, int n_steps, double dt_fs, int n_kpm, std::string axis, std::string method, int n_kpm_compute) {
             std::vector<double> msd_out;
             std::vector<double> energy_out;
             std::vector<double> time_out;
             self.compute_msd_spectrum(n_steps, dt_fs, n_kpm, msd_out, energy_out, time_out, axis,
                                       method, n_kpm_compute);
             return py::make_tuple(msd_out, energy_out, time_out);
         },
         py::arg("n_steps"),
         py::arg("dt_fs"),
         py::arg("n_kpm"),
         py::arg("axis") = "x",
         py::arg("method") = "auto",
         py::arg("n_kpm_compute") = 0)
        .def("compute_quantum_metric", [](rsatb::RSATB& self, int nt, int n_kpm, double temp, double mu, std::string axis, double omega_min_ev) {
             std::vector<double> gxx_out;
             std::vector<double> sigma_xx_out;
             std::vector<double> omega_out;
             self.compute_quantum_metric(nt, n_kpm, temp, mu, gxx_out, sigma_xx_out, omega_out, axis, omega_min_ev);
             return py::make_tuple(gxx_out, sigma_xx_out, omega_out);
         },
         py::arg("nt"),
         py::arg("n_kpm"),
         py::arg("temp"),
         py::arg("mu"),
         py::arg("axis") = "abc",
         py::arg("omega_min_ev") = 0.0)
        .def("atomic_data", &rsatb::RSATB::atomic_data, py::return_value_policy::reference);

}

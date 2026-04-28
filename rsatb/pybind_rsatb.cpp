#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include "rsatb.hpp"

namespace py = pybind11;

namespace {

rsatb::backend::SKRadialFunction parse_sktb_radial_function(py::handle handle) {
    py::dict cfg = py::cast<py::dict>(handle);
    rsatb::backend::SKRadialFunction radial;
    if (cfg.contains("kind")) radial.kind = py::cast<std::string>(cfg["kind"]);
    if (cfg.contains("r")) radial.r = py::cast<std::vector<double>>(cfg["r"]);
    if (cfg.contains("distances")) radial.r = py::cast<std::vector<double>>(cfg["distances"]);
    if (cfg.contains("values")) radial.values = py::cast<std::vector<double>>(cfg["values"]);
    if (cfg.contains("coeffs")) radial.coeffs = py::cast<std::vector<double>>(cfg["coeffs"]);
    if (cfg.contains("lambda")) radial.lambda_value = py::cast<double>(cfg["lambda"]);
    if (cfg.contains("r_on")) radial.r_on = py::cast<double>(cfg["r_on"]);
    if (cfg.contains("r_cut")) radial.r_cut = py::cast<double>(cfg["r_cut"]);
    return radial;
}

std::map<std::string, double> parse_sktb_onsite_map(py::handle handle) {
    py::dict cfg = py::cast<py::dict>(handle);
    std::map<std::string, double> result;
    for (const auto& item : cfg) {
        result.emplace(py::cast<std::string>(item.first), py::cast<double>(item.second));
    }
    return result;
}

rsatb::backend::SKPairInput parse_sktb_pair(py::handle handle) {
    py::dict cfg = py::cast<py::dict>(handle);
    rsatb::backend::SKPairInput pair;
    if (!cfg.contains("species")) {
        throw std::runtime_error("SKTB pair entry requires a 'species' list.");
    }
    const auto species = py::cast<std::vector<std::string>>(cfg["species"]);
    if (species.size() != 2) {
        throw std::runtime_error("SKTB pair entry requires exactly two species labels.");
    }
    pair.species[0] = species[0];
    pair.species[1] = species[1];

    if (cfg.contains("channels")) {
        py::dict channels = py::cast<py::dict>(cfg["channels"]);
        for (const auto& item : channels) {
            pair.channels.emplace(
                py::cast<std::string>(item.first),
                parse_sktb_radial_function(item.second));
        }
    }
    return pair;
}

rsatb::backend::SKMatrixInput parse_sktb_matrix_input(py::handle handle) {
    py::dict cfg = py::cast<py::dict>(handle);
    rsatb::backend::SKMatrixInput model;
    if (cfg.contains("onsite")) {
        py::dict onsite = py::cast<py::dict>(cfg["onsite"]);
        for (const auto& item : onsite) {
            model.onsite.emplace(
                py::cast<std::string>(item.first),
                parse_sktb_onsite_map(item.second));
        }
    }
    if (cfg.contains("pairs")) {
        const auto pairs = py::cast<std::vector<py::dict>>(cfg["pairs"]);
        model.pairs.reserve(pairs.size());
        for (const auto& pair : pairs) {
            model.pairs.push_back(parse_sktb_pair(pair));
        }
    }
    return model;
}

rsatb::backend::SKTBInput parse_sktb_input(py::handle handle) {
    py::dict cfg = py::cast<py::dict>(handle);
    rsatb::backend::SKTBInput source;

    if (cfg.contains("basis")) {
        py::dict basis = py::cast<py::dict>(cfg["basis"]);
        for (const auto& item : basis) {
            source.basis.emplace(
                py::cast<std::string>(item.first),
                py::cast<std::vector<std::string>>(item.second));
        }
    }
    if (cfg.contains("species_order")) {
        throw std::runtime_error(
            "SKTB setup no longer accepts sources.sktb.species_order. "
            "Provide type_symbols when constructing RSATB instead.");
    }
    if (cfg.contains("type_symbols")) {
        throw std::runtime_error(
            "SKTB setup does not accept sources.sktb.type_symbols. "
            "Provide type_symbols when constructing RSATB instead.");
    }
    if (cfg.contains("hamiltonian")) {
        source.hamiltonian = parse_sktb_matrix_input(cfg["hamiltonian"]);
    }
    if (cfg.contains("overlap")) {
        source.overlap = parse_sktb_matrix_input(cfg["overlap"]);
    }

    return source;
}

rsatb::backend::SKTBModel parse_sktb_model(py::handle handle, const std::vector<std::string>& type_symbols) {
    return rsatb::backend::build_sktb_model(parse_sktb_input(handle), type_symbols);
}

rsatb::model::Hdf5Input parse_hdf5_input(py::handle handle) {
    py::dict cfg = py::cast<py::dict>(handle);
    rsatb::model::Hdf5Input input;
    if (cfg.contains("filename")) input.filename = py::cast<std::string>(cfg["filename"]);
    if (cfg.contains("lspinorb")) input.lspinorb = py::cast<bool>(cfg["lspinorb"]);
    if (cfg.contains("nspin")) input.nspin = py::cast<int>(cfg["nspin"]);
    if (cfg.contains("noncolin")) input.noncolin = py::cast<bool>(cfg["noncolin"]);
    return input;
}

rsatb::model::LCAOIntegralInput parse_lcao_integrals(py::handle handle) {
    py::dict cfg = py::cast<py::dict>(handle);
    rsatb::model::LCAOIntegralInput integrals;
    if (cfg.contains("orb_names")) integrals.orb_names = py::cast<std::vector<std::string>>(cfg["orb_names"]);
    if (cfg.contains("orb_dir")) integrals.orb_dir = py::cast<std::string>(cfg["orb_dir"]);
    if (cfg.contains("psd_names")) integrals.psd_names = py::cast<std::vector<std::string>>(cfg["psd_names"]);
    if (cfg.contains("psd_dir")) integrals.psd_dir = py::cast<std::string>(cfg["psd_dir"]);
    if (cfg.contains("nspin")) integrals.nspin = py::cast<int>(cfg["nspin"]);
    if (cfg.contains("lspinorb")) integrals.lspinorb = py::cast<bool>(cfg["lspinorb"]);
    if (cfg.contains("noncolin")) integrals.noncolin = py::cast<bool>(cfg["noncolin"]);
    if (cfg.contains("log_file")) integrals.log_file = py::cast<std::string>(cfg["log_file"]);
    return integrals;
}

rsatb::model::SetupOptions parse_setup_options(py::handle handle) {
    rsatb::model::SetupOptions options;
    if (handle.is_none()) {
        return options;
    }

    py::dict cfg = py::cast<py::dict>(handle);
    if (cfg.contains("kpoints")) {
        options.kpoints = py::cast<std::vector<std::vector<double>>>(cfg["kpoints"]);
    }
    if (cfg.contains("nsamples")) {
        options.nsamples = py::cast<int>(cfg["nsamples"]);
    }
    if (cfg.contains("compute_bounds")) {
        options.compute_bounds = py::cast<bool>(cfg["compute_bounds"]);
    }
    if (cfg.contains("preconditioner")) {
        py::dict pre = py::cast<py::dict>(cfg["preconditioner"]);
        if (pre.contains("mode")) options.preconditioner.mode = py::cast<std::string>(pre["mode"]);
        if (pre.contains("tol")) options.preconditioner.tol = py::cast<double>(pre["tol"]);
        if (pre.contains("sptol")) options.preconditioner.sptol = py::cast<double>(pre["sptol"]);
        if (pre.contains("maxiter")) options.preconditioner.maxiter = py::cast<int>(pre["maxiter"]);
        if (pre.contains("method")) options.preconditioner.method = py::cast<std::string>(pre["method"]);
    }
    return options;
}

rsatb::model::LCAOModelConfig parse_lcao_model(py::handle handle) {
    py::dict cfg = py::cast<py::dict>(handle);
    rsatb::model::LCAOModelConfig model;
    if (cfg.contains("input")) {
        model.input = parse_hdf5_input(cfg["input"]);
    }
    if (cfg.contains("integrals")) {
        model.integrals = parse_lcao_integrals(cfg["integrals"]);
    }
    return model;
}

template <typename ModelConfig>
ModelConfig parse_pairwise_model(py::handle handle, const std::vector<std::string>& type_symbols) {
    py::dict cfg = py::cast<py::dict>(handle);
    ModelConfig model;
    if (cfg.contains("input")) {
        model.input = parse_hdf5_input(cfg["input"]);
    }
    if (cfg.contains("native")) {
        model.native = parse_sktb_model(cfg["native"], type_symbols);
    } else if (cfg.contains("basis") || cfg.contains("hamiltonian") || cfg.contains("overlap")) {
        model.native = parse_sktb_model(handle, type_symbols);
    }
    return model;
}

rsatb::model::AnyModelConfig parse_model_config(py::handle handle, const std::vector<std::string>& type_symbols) {
    py::dict cfg = py::cast<py::dict>(handle);
    if (!cfg.contains("kind")) {
        throw std::runtime_error("setup_config requires a model.kind entry.");
    }

    const std::string kind = py::cast<std::string>(cfg["kind"]);
    if (kind == "lcao") {
        return rsatb::model::AnyModelConfig(parse_lcao_model(cfg));
    }
    if (kind == "sktb") {
        return rsatb::model::AnyModelConfig(parse_pairwise_model<rsatb::model::SKTBModelConfig>(cfg, type_symbols));
    }
    if (kind == "dftb") {
        return rsatb::model::AnyModelConfig(parse_pairwise_model<rsatb::model::DFTBModelConfig>(cfg, type_symbols));
    }
    throw std::runtime_error("Unsupported model.kind '" + kind + "'.");
}

py::dict pack_sktb_atomic_preparation(const rsatb::backend::SKTBAtomicPreparation& prepared) {
    py::dict result;
    result["type_symbols"] = prepared.type_symbols;
    result["type_norb"] = prepared.type_norb;
    result["r_max_per_type"] = prepared.r_max_per_type;
    return result;
}

} // namespace

PYBIND11_MODULE(_rsatb, m) {
    m.doc() = "RSATB Python Binding";

    py::class_<rsatb::RSATB>(m, "RSATB")
        .def(py::init([](int npar,
                         size_t n_atom, size_t N_atom, size_t atom_offset, size_t n_edge, size_t N_edge,
                         py::array_t<int> atom_index, py::array_t<int> atom_type, 
                         py::array_t<int> edge_index, py::array_t<int> type_norb, 
                         py::array_t<int> edge_shift_vec,
                         py::array_t<double> cell, py::array_t<double> pos, bool verbose) {
            return new rsatb::RSATB(npar, n_atom, N_atom, atom_offset, n_edge, N_edge,
                                    atom_index.data(), atom_type.data(), 
                                    edge_index.data(), type_norb.data(), 
                                    std::vector<std::string>{},
                                    edge_shift_vec.data(),
                                    cell.data(), pos.data(), verbose);
        }),
        py::arg("npar"), py::arg("n_atom"), py::arg("N_atom"), py::arg("atom_offset"), 
        py::arg("n_edge"), py::arg("N_edge"),
        py::arg("atom_index"), py::arg("atom_type"), py::arg("edge_index"), 
        py::arg("type_norb"), py::arg("edge_shift_vec"),
        py::arg("cell"), py::arg("pos"), py::arg("verbose") = true)
        .def(py::init([](int npar,
                         size_t n_atom, size_t N_atom, size_t atom_offset, size_t n_edge, size_t N_edge,
                         py::array_t<int> atom_index, py::array_t<int> atom_type,
                         py::array_t<int> edge_index, py::array_t<int> type_norb,
                         const std::vector<std::string>& type_symbols,
                         py::array_t<int> edge_shift_vec,
                         py::array_t<double> cell, py::array_t<double> pos, bool verbose) {
            return new rsatb::RSATB(npar, n_atom, N_atom, atom_offset, n_edge, N_edge,
                                    atom_index.data(), atom_type.data(),
                                    edge_index.data(), type_norb.data(),
                                    type_symbols,
                                    edge_shift_vec.data(),
                                    cell.data(), pos.data(), verbose);
        }),
        py::arg("npar"), py::arg("n_atom"), py::arg("N_atom"), py::arg("atom_offset"),
        py::arg("n_edge"), py::arg("N_edge"),
        py::arg("atom_index"), py::arg("atom_type"), py::arg("edge_index"),
        py::arg("type_norb"), py::arg("type_symbols"), py::arg("edge_shift_vec"),
        py::arg("cell"), py::arg("pos"), py::arg("verbose") = true)
        .def(py::init<int, const std::string&, const std::vector<double>&, const std::vector<int>&, bool>(),
             py::arg("npar"), py::arg("filename"), py::arg("r_max_per_type"), py::arg("type_norb"), py::arg("verbose") = true)
        .def(py::init<int, const std::vector<double>&, const std::vector<int>&, 
                      const std::vector<double>&, const std::vector<bool>&, 
                      const std::vector<double>&, const std::vector<int>&, bool>(),
             py::arg("npar"), py::arg("pos"), py::arg("z"), py::arg("cell"), py::arg("pbc"),
             py::arg("r_max_per_type"), py::arg("type_norb"), py::arg("verbose") = true)
        .def("get_kernel_info", &rsatb::RSATB::get_kernel_info)
        .def("setup_model", [](rsatb::RSATB& self, py::dict model_cfg, py::dict options_cfg) {
             self.setup(parse_model_config(model_cfg, self.type_symbols()), parse_setup_options(options_cfg));
         },
         py::arg("model"),
         py::arg("options") = py::dict())
        .def("setup_lcao", [](rsatb::RSATB& self, py::dict model_cfg, py::dict options_cfg) {
             self.setup(parse_lcao_model(model_cfg), parse_setup_options(options_cfg));
         },
         py::arg("model"),
         py::arg("options") = py::dict())
        .def("setup_sktb", [](rsatb::RSATB& self, py::dict model_cfg, py::dict options_cfg) {
             self.setup(parse_pairwise_model<rsatb::model::SKTBModelConfig>(model_cfg, self.type_symbols()), parse_setup_options(options_cfg));
         },
         py::arg("model"),
         py::arg("options") = py::dict())
        .def("setup_dftb", [](rsatb::RSATB& self, py::dict model_cfg, py::dict options_cfg) {
             self.setup(parse_pairwise_model<rsatb::model::DFTBModelConfig>(model_cfg, self.type_symbols()), parse_setup_options(options_cfg));
         },
         py::arg("model"),
         py::arg("options") = py::dict())
        .def("setup_config", [](rsatb::RSATB& self, py::dict cfg) {
             if (cfg.contains("disorder")) {
                 throw std::runtime_error(
                     "setup_config no longer accepts 'disorder'; "
                     "apply disorder as a later post-processing step on initialized operators."
                 );
             }
             if (!cfg.contains("model")) {
                 throw std::runtime_error("setup_config requires a top-level 'model' block.");
             }
             self.setup(parse_model_config(cfg["model"], self.type_symbols()), parse_setup_options(cfg));
         }, py::arg("config"))
         .def("test_solve", &rsatb::RSATB::test_solve)
        .def("compute_dos", [](rsatb::RSATB& self, int nt) {
             std::vector<double> dos_out;
             std::vector<double> omega_out;
             self.compute_dos(nt, dos_out, omega_out);
             return std::make_pair(dos_out, omega_out);
         }, py::arg("nt"))
        .def("compute_ac", [](rsatb::RSATB& self, int nt, int n_kpm, double temp, double mu, std::string axis) {
             std::vector<double> ac_out;
             std::vector<double> omega_out;
             self.compute_ac(nt, n_kpm, temp, mu, ac_out, omega_out, axis);
             return std::make_pair(ac_out, omega_out);
         }, py::arg("nt"), py::arg("n_kpm"), py::arg("temp"), py::arg("mu"), py::arg("axis") = "abc")
        .def("compute_dc_spectrum", [](rsatb::RSATB& self, int n_moments, std::string formulation, std::string axis, int partition_sections, size_t max_workspace_mb) {
             std::vector<double> sigma_out;
             std::vector<double> energy_out;
             self.compute_dc_spectrum(n_moments, sigma_out, energy_out, formulation, axis, partition_sections, max_workspace_mb);
             return std::make_pair(sigma_out, energy_out);
         },
         py::arg("n_moments"),
         py::arg("formulation") = "greenwood",
         py::arg("axis") = "abc",
         py::arg("partition_sections") = 0,
         py::arg("max_workspace_mb") = 0)
        .def("compute_dc", [](rsatb::RSATB& self, int n_moments, double mu, double temp, std::string formulation, std::string axis, int partition_sections, size_t max_workspace_mb) {
             std::vector<double> dc_out;
             self.compute_dc(n_moments, mu, temp, dc_out, formulation, axis, partition_sections, max_workspace_mb);
             return dc_out;
         },
         py::arg("n_moments"),
         py::arg("mu"),
         py::arg("temp"),
         py::arg("formulation") = "greenwood",
         py::arg("axis") = "abc",
         py::arg("partition_sections") = 0,
         py::arg("max_workspace_mb") = 0)
        .def("compute_msd_spectrum", [](rsatb::RSATB& self, int n_steps, double dt_fs, int n_kpm, std::string axis) {
             std::vector<double> msd_out;
             std::vector<double> energy_out;
             std::vector<double> time_out;
             self.compute_msd_spectrum(n_steps, dt_fs, n_kpm, msd_out, energy_out, time_out, axis);
             return py::make_tuple(msd_out, energy_out, time_out);
         },
         py::arg("n_steps"),
         py::arg("dt_fs"),
         py::arg("n_kpm"),
         py::arg("axis") = "x")
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

    m.def("_prepare_sktb_atomic_spec_from_file",
          [](const std::string& filename, py::dict sktb_cfg) {
              const auto source = parse_sktb_input(sktb_cfg);
              const auto structure = rsatb::backend::io::read_structure(filename);
              return pack_sktb_atomic_preparation(
                  rsatb::backend::prepare_sktb_atomic_data_from_atomic_numbers(structure.z, source));
          },
          py::arg("filename"), py::arg("sktb_config"));

    m.def("_prepare_sktb_atomic_spec_from_atomic_numbers",
          [](const std::vector<int>& z, py::dict sktb_cfg) {
              const auto source = parse_sktb_input(sktb_cfg);
              return pack_sktb_atomic_preparation(
                  rsatb::backend::prepare_sktb_atomic_data_from_atomic_numbers(z, source));
          },
          py::arg("atomic_numbers"), py::arg("sktb_config"));
}

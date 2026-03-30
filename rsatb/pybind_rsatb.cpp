#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include "rsatb.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_rsatb, m) {
    m.doc() = "RSATB Python Binding";
    using LegacySetup = void (rsatb::RSATB::*)(
        const std::string&,
        const std::vector<std::vector<double>>&,
        const std::vector<std::string>&,
        const std::string&,
        const std::vector<std::string>&,
        const std::string&,
        double,
        double,
        int,
        const std::string&,
        const std::string&,
        int,
        bool
    );

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
                                    edge_shift_vec.data(),
                                    cell.data(), pos.data(), verbose);
        }),
        py::arg("npar"), py::arg("n_atom"), py::arg("N_atom"), py::arg("atom_offset"), 
        py::arg("n_edge"), py::arg("N_edge"),
        py::arg("atom_index"), py::arg("atom_type"), py::arg("edge_index"), 
        py::arg("type_norb"), py::arg("edge_shift_vec"),
        py::arg("cell"), py::arg("pos"), py::arg("verbose") = true)
        .def(py::init<int, const std::string&, const std::vector<double>&, const std::vector<int>&, bool>(),
             py::arg("npar"), py::arg("filename"), py::arg("r_max_per_type"), py::arg("type_norb"), py::arg("verbose") = true)
        .def(py::init<int, const std::vector<double>&, const std::vector<int>&, 
                      const std::vector<double>&, const std::vector<bool>&, 
                      const std::vector<double>&, const std::vector<int>&, bool>(),
             py::arg("npar"), py::arg("pos"), py::arg("z"), py::arg("cell"), py::arg("pbc"),
             py::arg("r_max_per_type"), py::arg("type_norb"), py::arg("verbose") = true)
        .def("get_kernel_info", &rsatb::RSATB::get_kernel_info)
        .def("setup", static_cast<LegacySetup>(&rsatb::RSATB::setup),
             py::arg("Hfile"), py::arg("kpoints"),
             py::arg("orb_names") = std::vector<std::string>{}, py::arg("orb_dirs") = "",
             py::arg("psd_names") = std::vector<std::string>{}, py::arg("psd_dirs") = "",
             py::arg("tol") = 1e-7, py::arg("sptol") = 1e-5, py::arg("maxiter")=20, py::arg("method")="graph",
             py::arg("precond") = "H/S", py::arg("nsamples") = 1, py::arg("compute_bounds") = true)
        .def("setup_config", [](rsatb::RSATB& self, py::dict cfg) {
             rsatb::backend::OperatorSetupConfig config;
             int nsamples = cfg.contains("nsamples") ? py::cast<int>(cfg["nsamples"]) : 1;
             if (cfg.contains("kpoints")) {
                 config.kpoints = py::cast<std::vector<std::vector<double>>>(cfg["kpoints"]);
             }
             if (config.kpoints.empty()) {
                 config.kpoints = {{0.0, 0.0, 0.0}};
             }
             if (cfg.contains("compute_bounds")) {
                 config.compute_bounds = py::cast<bool>(cfg["compute_bounds"]);
             }

             if (cfg.contains("sources")) {
                 py::dict sources = py::cast<py::dict>(cfg["sources"]);
                 if (sources.contains("hdf5")) {
                     py::dict hdf5 = py::cast<py::dict>(sources["hdf5"]);
                     if (hdf5.contains("filename")) config.sources.hdf5.filename = py::cast<std::string>(hdf5["filename"]);
                     if (hdf5.contains("lspinorb")) config.sources.hdf5.lspinorb = py::cast<bool>(hdf5["lspinorb"]);
                     if (hdf5.contains("nspin")) config.sources.hdf5.nspin = py::cast<int>(hdf5["nspin"]);
                     if (hdf5.contains("noncolin")) config.sources.hdf5.noncolin = py::cast<bool>(hdf5["noncolin"]);
                 }
                 if (sources.contains("op2c")) {
                     py::dict op2c = py::cast<py::dict>(sources["op2c"]);
                     if (op2c.contains("orb_names")) config.sources.op2c.orb_names = py::cast<std::vector<std::string>>(op2c["orb_names"]);
                     if (op2c.contains("orb_dir")) config.sources.op2c.orb_dir = py::cast<std::string>(op2c["orb_dir"]);
                     if (op2c.contains("psd_names")) config.sources.op2c.psd_names = py::cast<std::vector<std::string>>(op2c["psd_names"]);
                     if (op2c.contains("psd_dir")) config.sources.op2c.psd_dir = py::cast<std::string>(op2c["psd_dir"]);
                     if (op2c.contains("nspin")) config.sources.op2c.nspin = py::cast<int>(op2c["nspin"]);
                     if (op2c.contains("lspinorb")) config.sources.op2c.lspinorb = py::cast<bool>(op2c["lspinorb"]);
                     if (op2c.contains("noncolin")) config.sources.op2c.noncolin = py::cast<bool>(op2c["noncolin"]);
                     if (op2c.contains("log_file")) config.sources.op2c.log_file = py::cast<std::string>(op2c["log_file"]);
                 }
             }

             if (cfg.contains("preconditioner")) {
                 py::dict pre = py::cast<py::dict>(cfg["preconditioner"]);
                 if (pre.contains("mode")) config.preconditioner.mode = py::cast<std::string>(pre["mode"]);
                 if (pre.contains("tol")) config.preconditioner.tol = py::cast<double>(pre["tol"]);
                 if (pre.contains("sptol")) config.preconditioner.sptol = py::cast<double>(pre["sptol"]);
                 if (pre.contains("maxiter")) config.preconditioner.maxiter = py::cast<int>(pre["maxiter"]);
                 if (pre.contains("method")) config.preconditioner.method = py::cast<std::string>(pre["method"]);
             }

             if (cfg.contains("disorder")) {
                 py::dict disorder = py::cast<py::dict>(cfg["disorder"]);
                 if (disorder.contains("kind")) config.disorder.kind = py::cast<std::string>(disorder["kind"]);
                 if (disorder.contains("strength_ev")) config.disorder.strength_ev = py::cast<double>(disorder["strength_ev"]);
                 if (disorder.contains("seed")) config.disorder.seed = py::cast<std::uint64_t>(disorder["seed"]);
                 if (disorder.contains("per_atom")) config.disorder.per_atom = py::cast<bool>(disorder["per_atom"]);
             }

             self.setup(config, nsamples);
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
}

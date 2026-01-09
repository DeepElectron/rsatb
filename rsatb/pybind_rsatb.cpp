#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include "rsatb.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_rsatb, m) {
    m.doc() = "RSATB Python Binding";

    py::class_<rsatb::RSATB>(m, "RSATB")
        .def(py::init([](int npar,
                         size_t n_atom, size_t N_atom, size_t atom_offset, size_t n_edge, size_t N_edge,
                         py::array_t<int> atom_index, py::array_t<int> atom_type, 
                         py::array_t<int> edge_index, py::array_t<int> type_norb, 
                         py::array_t<int> edge_shift_vec,
                         py::array_t<double> cell, py::array_t<double> pos) {
            return new rsatb::RSATB(npar, n_atom, N_atom, atom_offset, n_edge, N_edge,
                                    atom_index.data(), atom_type.data(), 
                                    edge_index.data(), type_norb.data(), 
                                    edge_shift_vec.data(),
                                    cell.data(), pos.data());
        }),
        py::arg("npar"), py::arg("n_atom"), py::arg("N_atom"), py::arg("atom_offset"), 
        py::arg("n_edge"), py::arg("N_edge"),
        py::arg("atom_index"), py::arg("atom_type"), py::arg("edge_index"), 
        py::arg("type_norb"), py::arg("edge_shift_vec"),
        py::arg("cell"), py::arg("pos"))
        .def(py::init<int, const std::string&, const std::vector<double>&, const std::vector<int>&>(),
             py::arg("npar"), py::arg("filename"), py::arg("r_max_per_type"), py::arg("type_norb"))
        .def(py::init<int, const std::vector<double>&, const std::vector<int>&, 
                      const std::vector<double>&, const std::vector<bool>&, 
                      const std::vector<double>&, const std::vector<int>&>(),
             py::arg("npar"), py::arg("pos"), py::arg("z"), py::arg("cell"), py::arg("pbc"),
             py::arg("r_max_per_type"), py::arg("type_norb"))
        .def("get_kernel_info", &rsatb::RSATB::get_kernel_info)
        .def("setup", &rsatb::RSATB::setup,
             py::arg("Hfile"), py::arg("kpoints"),
             py::arg("orb_names") = std::vector<std::string>{}, py::arg("orb_dirs") = "",
             py::arg("psd_names") = std::vector<std::string>{}, py::arg("psd_dirs") = "",
             py::arg("precond") = "H/S", py::arg("nsamples") = 1)
        .def("compute_dos", [](rsatb::RSATB& self, int nt) {
             std::vector<double> dos_out;
             std::vector<double> omega_out;
             self.compute_dos(nt, dos_out, omega_out);
             return std::make_pair(dos_out, omega_out);
         }, py::arg("nt"))
        .def("compute_ac", [](rsatb::RSATB& self, int nt, int n_kpm, double temp, double mu) {
             std::vector<double> ac_out;
             std::vector<double> omega_out;
             self.compute_ac(nt, n_kpm, temp, mu, ac_out, omega_out);
             return std::make_pair(ac_out, omega_out);
         }, py::arg("nt"), py::arg("n_kpm"), py::arg("temp"), py::arg("mu"))
        .def("get_operator_construction", &rsatb::RSATB::get_operator_construction, py::return_value_policy::reference)
        .def("atomic_data", &rsatb::RSATB::atomic_data, py::return_value_policy::reference);
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include "rsatb.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_rsatb, m) {
    m.doc() = "RSATB Python Binding";

    py::class_<rsatb::RSATB>(m, "RSATB")
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
        .def("compute_dos", &rsatb::RSATB::compute_dos,
             py::arg("nt"), py::arg("dos_out"), py::arg("omega_out"))
        .def("compute_ac", &rsatb::RSATB::compute_ac,
             py::arg("nt"), py::arg("n_kpm"), py::arg("temp"), py::arg("mu"),
             py::arg("ac_out"), py::arg("omega_out"));
}

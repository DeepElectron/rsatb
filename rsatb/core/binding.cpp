#include "solver.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // if you bind std::vector
#include <pybind11/complex.h>   // if you bind std::complex
#include <pybind11/numpy.h>
#include <mpi4py/mpi4py.h>
#include <mpi.h>
#include <tuple>
#include <iostream>

using namespace pybind11::literals;
using namespace rsatb;
namespace py = pybind11;

MPI_Comm *get_mpi_comm(py::object py_comm) {
  auto comm_ptr = PyMPIComm_Get(py_comm.ptr());

  if (!comm_ptr)
    throw py::error_already_set();

  return comm_ptr;
}


PYBIND11_DECLARE_HOLDER_TYPE(T, std::unique_ptr<T>, true);
PYBIND11_MODULE(binding, m) {
    // initialize mpi4py's C-API
    if (import_mpi4py() < 0) {
        // mpi4py calls the Python C API
        // we let pybind11 give us the detailed traceback
        throw py::error_already_set();
    }
    py::class_<rsatb::Solver>(m, "Solver")
        .def(py::init([](
            std::string &HFile, std::string &SFile,
            py::array_t<int, py::array::c_style | py::array::forcecast> atom_type, 
            py::array_t<double, py::array::c_style | py::array::forcecast> pos,
            py::array_t<int, py::array::c_style | py::array::forcecast> edge_index, 
            py::array_t<int, py::array::c_style | py::array::forcecast> type_norb, 
            py::array_t<int, py::array::c_style | py::array::forcecast> edge_shift_vec,
            py::object py_comm) {

                // Get the buffers
                py::buffer_info at_bf = atom_type.request(), ei_bf = edge_index.request(), tn_bf = type_norb.request();
                py::buffer_info esv_bf = edge_shift_vec.request();
                py::buffer_info pos_bf = pos.request();

                /*
                Do checks:
                    1. Dim check, atom_type [n_atom], pos [n_atom, 3], edge_index [n_edge, 2], type_norb [n_type], edge_shift_vec [n_edge, 3]
                    2. Transform pycomm to c++ comm;
                    3. Convert the buff to pointers with type conversion (if type conversion is not forces in the function annotation. Here we do not need to);
                */
            
                // 1.
                if (at_bf.ndim != 1)
                    throw std::runtime_error("atom_type must be a 1D array of length n_atom");
                size_t n_atom = at_bf.size;
                if (pos_bf.ndim != 2 || pos_bf.size != 3*n_atom)
                    throw std::runtime_error("pos must be 2D of shape [n_atom, 3]");
                if (ei_bf.ndim != 2)
                    throw std::runtime_error("edge_index must be 2D of length 2*n_edge");
                size_t n_edge = ei_bf.size / 2;
                if (tn_bf.ndim != 1)
                    throw std::runtime_error("type_norb must be 1D");
                if (esv_bf.ndim != 2 || esv_bf.size != 3*n_edge)
                    throw std::runtime_error("edge_shift_vec must be 2D with shape [n_edge, 3]");

                // 2.
                MPI_Comm *comm = get_mpi_comm(py_comm); // what we get is the ptr to Comm;

                // 3. Skip
                std::cout << "n_edge: " << n_edge << "\n";
                std::cout << "n_edge: " << n_atom << "\n";
                for(int i=0;i<n_atom;++i) std::cout<<atom_type.data()[i]<<" ";
                std::cout<<"\n";
                for(int i=0;i<n_edge;++i) std::cout<<edge_index.data()[i]<<" ";
                std::cout<<"\n";

                return std::make_unique<rsatb::Solver>(HFile, SFile, n_atom, n_edge, 
                    atom_type.data(), edge_index.data(), type_norb.data(), edge_shift_vec.data(), pos.data(), *comm);

        }),
        py::keep_alive<1,4>(),
        py::keep_alive<1,5>(),
        py::keep_alive<1,6>(),
        py::keep_alive<1,7>(),
        py::keep_alive<1,8>() 
        )
        .def("setup_graph", &rsatb::Solver::setup_graph)

        .def("setup_HS", [](
            rsatb::Solver &solver, // Since python call self as the first arg
            py::array_t<double, py::array::c_style | py::array::forcecast> kpoint
        ){
            py::buffer_info kp_bf = kpoint.request();
            if (kp_bf.size != 3)
                throw std::runtime_error("kpoint format error");
            solver.setup_HS(kpoint.data()[0], kpoint.data()[1], kpoint.data()[2]);
        })

        .def("setup_velocityOp", &rsatb::Solver::setup_velocityOp)

        .def("setup_sample", &rsatb::Solver::setup_sample)

        .def("get_nt", [](
            rsatb::Solver &solver,
            double dE
        ){
            int nt;
            solver.get_nt(dE, &nt);
            return nt;
        })

        .def("compute_dos", [](
            rsatb::Solver &solver,
            int nt,
            std::string proj_mode,
            py::array_t<int, py::array::c_style | py::array::forcecast> orbtype
        ){
            

            double dos[2*nt+1];
            std::unique_ptr<double[]> pdos;
            int n_proj;

            solver.compute_dos(nt, proj_mode, orbtype.data(), &n_proj, dos, pdos);

            py::array_t<double> o_dos(2*nt+1);
            py::array_t<double> o_pdos((2*nt+1)*n_proj);

            if(solver.rank == 0){
            
                std::memcpy(o_dos.mutable_data(), dos, (2*nt+1) * sizeof(double));
                
                if (proj_mode != "none"){
                    std::memcpy(o_pdos.mutable_data(), pdos.get(), n_proj * (2*nt+1) * sizeof(double));
                }
            }
            
            return std::make_tuple(o_dos, o_pdos);
        })

        .def("__repr__", [](rsatb::Solver &a){
            return "<rsatb.Solver c++ driver>";
        });
}
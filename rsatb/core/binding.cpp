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
                py::array_t<int, py::array::c_style | py::array::forcecast> atom_index,
                py::array_t<int, py::array::c_style | py::array::forcecast> atom_type, 
                py::array_t<double, py::array::c_style | py::array::forcecast> cell,
                py::array_t<double, py::array::c_style | py::array::forcecast> pos,
                py::array_t<int, py::array::c_style | py::array::forcecast> edge_index, 
                py::array_t<int, py::array::c_style | py::array::forcecast> type_norb, 
                py::array_t<int, py::array::c_style | py::array::forcecast> edge_shift_vec,
                py::object py_comm, std::string options
            ){
                // Get the buffers
                py::buffer_info ai_bf = atom_index.request(), at_bf = atom_type.request(), ei_bf = edge_index.request(), tn_bf = type_norb.request();
                py::buffer_info esv_bf = edge_shift_vec.request(), cell_bf = cell.request();
                py::buffer_info pos_bf = pos.request();

                /*
                Do checks:
                    1. Dim check, atom_type [n_atom], pos [n_atom, 3], edge_index [n_edge, 2], type_norb [n_type], edge_shift_vec [n_edge, 3]
                    2. Transform pycomm to c++ comm;
                    3. Convert the buff to pointers with type conversion (if type conversion is not forces in the function annotation. Here we do not need to);
                */
            
                // 1.
                if (ai_bf.ndim != 1)
                    throw std::runtime_error("atom_index must be a 1D array of length n_atom");
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
                if (cell_bf.ndim != 2 || cell_bf.size != 9)
                    throw std::runtime_error("cell must be 2D with shape [3, 3]");

                // 2.
                MPI_Comm *comm = get_mpi_comm(py_comm); // what we get is the ptr to Comm;
                size_t N_atom=n_atom, N_edge=n_edge, atom_offset=0;

                MPI_Allreduce(MPI_IN_PLACE, &N_atom, 1, MPI_INT, MPI_SUM, *comm);
                MPI_Allreduce(MPI_IN_PLACE, &N_edge, 1, MPI_INT, MPI_SUM, *comm);
                MPI_Exscan(&n_atom, &atom_offset, 1, MPI_INT, MPI_SUM, *comm);

                // 3. Skip
                // std::cout << "n_edge: " << n_edge << "\n";
                // std::cout << "n_edge: " << n_atom << "\n";
                // for(int i=0;i<n_atom;++i) std::cout<<atom_type.data()[i]<<" ";
                // std::cout<<"\n";
                // for(int i=0;i<n_edge;++i) std::cout<<edge_index.data()[i]<<" ";
                // std::cout<<"\n";

                return std::make_unique<rsatb::Solver>(n_atom, N_atom, atom_offset, n_edge, N_edge,
                    atom_index.data(), atom_type.data(), edge_index.data(), type_norb.data(), edge_shift_vec.data(), 
                    cell.data(), pos.data(), *comm, options);
            }),
            py::keep_alive<1,2>(),
            py::keep_alive<1,3>(),
            py::keep_alive<1,4>(),
            py::keep_alive<1,5>(),
            py::keep_alive<1,6>(),
            py::keep_alive<1,7>(),
            py::keep_alive<1,8>(),
            py::arg("atom_index"),
            py::arg("atom_type"),
            py::arg("cell"),
            py::arg("pos"),
            py::arg("edge_index"),
            py::arg("type_norb"),
            py::arg("edge_shift_vec"),
            py::arg("py_comm"),
            py::arg("options")=" "
        )

        .def("setup_K", &rsatb::Solver::setup_K)

        .def("setup_S_h5", [](rsatb::Solver &solver, const std::string &SFile)
            {
                solver.setup_S(SFile);
            },
            py::arg("SFile")
        )

        .def("setup_S_orb", [](
            rsatb::Solver &solver, 
            std::vector<std::string> file_orb0, 
            const std::string& orbital_dir
            ){
                if (file_orb0.size() != solver.ntype())
                    throw std::runtime_error("The length of the input orbitals does not match the number of chemical elements.");
                solver.setup_S(file_orb0.data(), orbital_dir);
            },
            py::arg("file_orb0"),
            py::arg("orbital_dir") = "./"
        )

        .def("test_Sinv", &rsatb::Solver::test_Sinv)

        .def("Ssqrtm", &rsatb::Solver::Ssqrtm, py::arg("tol")=1e-7, py::arg("sptol")=1e-6, py::arg("maxit")=20, py::arg("verbose")=false, py::arg("orthogonalize")=false)

        .def("setup_R_h5", [](
            rsatb::Solver &solver,
            const std::string &RxFile,
            const std::string &RyFile,
            const std::string &RzFile
            ){
                solver.setup_R(RxFile, RyFile, RzFile);
            },
            py::arg("RxFile")="",
            py::arg("RyFile")="",
            py::arg("RzFile")=""    
        )

        .def("setup_R_orb", [](
            rsatb::Solver &solver, 
            const std::vector<std::string> &file_orb0, 
            const std::string& orbital_dir
            ){
                solver.setup_R(file_orb0.data(), orbital_dir);
            },
            py::arg("file_orb0"),
            py::arg("orbital_dir") = "./"
        )

        .def("setup_H_h5", [](
            rsatb::Solver &solver, // Since python call self as the first arg
            const std::string &HFile
            ){
                solver.setup_H_h5(HFile);
            },
            py::arg("HFile")
        )
        .def("orthogonalize_H", &rsatb::Solver::orthogonalize_H)
        
        .def("setup_sample", &rsatb::Solver::setup_sample)

        .def("export_S", &rsatb::Solver::export_S)
        .def("export_H", &rsatb::Solver::export_H)
        .def("export_Rx", &rsatb::Solver::export_Rx)
        .def("export_Ry", &rsatb::Solver::export_Ry)
        .def("export_Rz", &rsatb::Solver::export_Rz)

        .def("scale_shift_H", &rsatb::Solver::scale_shift_H)

        .def("setup_KSP", &rsatb::Solver::setup_KSP)

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
            py::array_t<int, py::array::c_style | py::array::forcecast> orbtype,
            bool verbose
        ){  
            
            if (nt < 0) {
                throw std::runtime_error("The timestep cannot be negative!");
            }
            
            double dos[2*nt];
            double omegas[2*nt];
            std::unique_ptr<double[]> pdos;
            int n_proj=0;

            solver.compute_dos(nt, proj_mode, orbtype.data(), &n_proj, dos, pdos, omegas, verbose);

            py::array_t<double> o_dos(2*nt);
            py::array_t<double> o_omegas(2*nt);
            py::array_t<double> o_pdos((2*nt)*n_proj);

            if(solver.rank == 0){
            
                std::memcpy(o_dos.mutable_data(), dos, 2*nt * sizeof(double));
                std::memcpy(o_omegas.mutable_data(), omegas, 2*nt * sizeof(double));
                
                if (proj_mode != "none"){
                    std::memcpy(o_pdos.mutable_data(), pdos.get(), n_proj * 2*nt * sizeof(double));
                }
            }
            
            return std::make_tuple(o_dos, o_pdos, o_omegas);
        },
        py::keep_alive<1,2>(),
        py::keep_alive<1,3>(),
        py::keep_alive<1,4>(),
        py::arg("nt"),
        py::arg("proj_mode") = "none",
        py::arg("orbtype") = py::array_t<int>(),
        py::arg("verbose") = false
        )

        .def("compute_ac", [](
            rsatb::Solver &solver,
            int nt, int n_kpm,
            double temp, double mu
        ){
            double ac[6*(nt+1)], omegas[nt+1];
            solver.compute_ac(nt, n_kpm, temp, mu, ac, omegas);

            py::array_t<double> o_ac(6*(nt+1));
            py::array_t<double> o_omegas(nt+1);

            if(solver.rank == 0){
            
                std::memcpy(o_ac.mutable_data(), ac, 6*(nt+1) * sizeof(double));
                std::memcpy(o_omegas.mutable_data(), omegas, (nt+1) * sizeof(double));
                
            }

            return std::make_tuple(o_ac, o_omegas);
        },
        py::arg("nt"),
        py::arg("n_kpm"),
        py::arg("temp"),
        py::arg("mu")
        )

        .def("__repr__", [](rsatb::Solver &a){
            return "<rsatb.Solver c++ driver>";
        });
}
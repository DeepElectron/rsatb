#include "calc.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // if you bind std::vector
#include <pybind11/complex.h>   // if you bind std::complex
#include <pybind11/numpy.h>
#include <iostream>

using namespace pybind11::literals;
namespace py = pybind11;
using cd = std::complex<double>;

inline py::array_t<cd> tdos_bind(
                py::array_t<int, py::array::c_style | py::array::forcecast> h_indptr, 
                py::array_t<int, py::array::c_style | py::array::forcecast> h_indices, 
                py::array_t<cd, py::array::c_style | py::array::forcecast> h_data,
                py::array_t<cd, py::array::f_style | py::array::forcecast> Psi_in, // Psi should be with f style for convenience of copy? (n, nrhs)
                std::size_t n,
                std::size_t nrhs,
                py::array_t<double, py::array::c_style | py::array::forcecast> ts,
                py::array_t<int, py::array::c_style | py::array::forcecast> s_indptr=py::array_t<int>(),
                py::array_t<int, py::array::c_style | py::array::forcecast> s_indices=py::array_t<int>(),
                py::array_t<cd, py::array::c_style | py::array::forcecast> s_data=py::array_t<cd>(),
                bool forward=true,
                double btol=1e-20
) {
    py::buffer_info ih_bf = h_indptr.request(), jh_bf = h_indices.request(), h_bf = h_data.request();
    py::buffer_info ts_bf = ts.request();
    py::buffer_info Psi_in_bf = Psi_in.request();

    if (ih_bf.ndim != 1 || ih_bf.size != (n+1))
        throw std::runtime_error("ih must be a 1D array of length n+1");
    if (jh_bf.ndim != 1 || jh_bf.size != h_bf.size)
        throw std::runtime_error("jh must be 1D of length nnz");
    if (h_bf.ndim != 1)
        throw std::runtime_error("h must be 1D of length nnz");

    int *ih_ptr = static_cast<int *>(ih_bf.ptr);
    int *jh_ptr = static_cast<int *>(jh_bf.ptr);
    cd *h_ptr = static_cast<cd *>(h_bf.ptr);
    cd *Psi_in_ptr = static_cast<cd *>(Psi_in_bf.ptr);
    double *ts_ptr = static_cast<double *>(ts_bf.ptr);
    size_t ts_N = ts_bf.size;
    std::vector<double> ts_vec(ts_N);
    for(int i=0;i<ts_N;++i){
        ts_vec[i] = ts_ptr[i];
    }
    
    int *is_ptr = nullptr;
    int *js_ptr = nullptr;
    cd *s_ptr = nullptr;
    // auto result = py::array_t<double>(ts_bf.size);
    if (s_indptr.size() != 0 && s_indices.size() != 0 && s_data.size() != 0) {
        py::buffer_info is_bf = s_indptr.request(), js_bf = s_indices.request(), s_bf = s_data.request();
        is_ptr = static_cast<int *>(is_bf.ptr);
        js_ptr = static_cast<int *>(js_bf.ptr);
        s_ptr = static_cast<cd *>(s_bf.ptr);

        if (is_bf.ndim != 1 || is_bf.size != (n+1))
            throw std::runtime_error("is must be a 1D array of length n+1");
        if (js_bf.ndim != 1 || js_bf.size != s_bf.size)
            throw std::runtime_error("js must be 1D of length nnz");
        if (s_bf.ndim != 1)
            throw std::runtime_error("s must be 1D of length nnz");
    }

    std::vector<cd> result = tdos(
        ih_ptr, jh_ptr, h_ptr, Psi_in_ptr,
        n, nrhs, ts_vec,
        is_ptr, js_ptr, s_ptr,
        forward,
        btol
    );

    py::array_t<cd> out(result.size());
    auto out_bf = out.request();
    cd* out_ptr = static_cast<cd*>(out_bf.ptr);
    std::memcpy(out_ptr, result.data(), result.size() * sizeof(cd));

    return out;
}

PYBIND11_MODULE(rsatb, m) {
    m.doc() = "calculation module for rsatb"; // optional module docstring

    m.def("tdos", &tdos_bind, "time domain dos calculation", 
    "h_indptr"_a, "h_indices"_a, "h_data"_a, "Psi_in"_a, "n"_a, "nrhs"_a, "ts"_a, "s_indptr"_a=py::array_t<int>(), "s_indices"_a=py::array_t<int>(), "s_data"_a=py::array_t<cd>(), "forward"_a=true, "btol"_a=1e-20
    );
}
#ifndef DBCSRCAPI
#define DBCSRCAPI

#include <dbcsr.h>
#include <complex>
#include <vector>

inline double _Complex mk_c99(double re, double im) {
  double _Complex z{};
  __real__ z = re;
  __imag__ z = im;
  return z;
}

inline void c_dbcsr_put_block2d_z_api(dbcsr_matrix c_matrix, const int c_row, const int c_col, const std::complex<double> *c_block,
  const int c_row_size, const int c_col_size, const bool* c_summation, const std::complex<double> *c_scale){
    std::vector<double _Complex> tmp((size_t)(c_row_size*c_col_size));
    double _Complex stmp = mk_c99((*c_scale).real(), (*c_scale).imag());

    // for (size_t k = 0; k < tmp.size(); ++k) {
    //     tmp[k] = mk_c99(c_block[k].real(), c_block[k].imag());  // I from C99 is OK in gnu++.
    // }
    for (size_t ir=0; ir < c_row_size; ir++) {
      for (size_t ic=0; ic < c_col_size; ic++) {
        tmp[ic*c_row_size+ir] = mk_c99(c_block[ir*c_col_size+ic].real(), c_block[ir*c_col_size+ic].imag());
      }
    }

    c_dbcsr_put_block2d_z(c_matrix, c_row, c_col, tmp.data(), c_row_size, c_col_size, c_summation, &stmp);
}

inline void c_dbcsr_add_on_diag_z_api(dbcsr_matrix c_matrix, const std::complex<double> c_alpha_scalar){
    double _Complex stmp = mk_c99(c_alpha_scalar.real(), c_alpha_scalar.imag());
    c_dbcsr_add_on_diag_z(c_matrix, stmp);
}

#endif
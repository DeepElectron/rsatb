#ifndef FFTW
#define FFTW

#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <complex>

namespace rsatb::calc{

using cd = std::complex<double>;

void fft_1d(int n, cd *in, cd *out){
    fftw_plan p = fftw_plan_dft_1d(
        n, reinterpret_cast<fftw_complex*>(in), 
        reinterpret_cast<fftw_complex*>(out), 
        FFTW_FORWARD, FFTW_ESTIMATE
        );
    fftw_execute(p);
    fftw_destroy_plan(p);
    fftw_cleanup();
}

void ifft_1d(int n, cd *in, cd *out){
    fftw_plan p = fftw_plan_dft_1d(
        n, reinterpret_cast<fftw_complex*>(in), 
        reinterpret_cast<fftw_complex*>(out), 
        FFTW_BACKWARD, FFTW_ESTIMATE
        );

    fftw_execute(p);
    fftw_destroy_plan(p);
    fftw_cleanup();
}

// void bifft_id(){
    
// }

void bifft_1d(int n, int nb, cd* in, cd* out) {
    /*
        batched transform of size [n, nb] shaped array
    */
    int ns[1];         // FFT size
    ns[0] = n;

    fftw_plan p = fftw_plan_many_dft(
        1, ns, nb,
        reinterpret_cast<fftw_complex*>(in), nullptr,
        nb, 1,
        reinterpret_cast<fftw_complex*>(out), nullptr,
        nb, 1,
        FFTW_BACKWARD, FFTW_ESTIMATE
    );

    fftw_execute(p);
    fftw_destroy_plan(p);
    fftw_cleanup();
}

void bfft_1d(int n, int nb, cd* in, cd* out) {
    /*
        batched transform of size [nb, n] shaped array
    */
    int ns[1];         // FFT size
    ns[0] = n;

    fftw_plan p = fftw_plan_many_dft(
        1, ns, nb,
        reinterpret_cast<fftw_complex*>(in), nullptr,
        nb, 1,
        reinterpret_cast<fftw_complex*>(out), nullptr,
        nb, 1,
        FFTW_FORWARD, FFTW_ESTIMATE
    );

    fftw_execute(p);
    fftw_destroy_plan(p);
    fftw_cleanup();
}

};

#endif
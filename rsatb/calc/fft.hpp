#ifndef RSATB_CALC_FFT_HPP
#define RSATB_CALC_FFT_HPP

#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <algorithm>
#include <cassert>
#include <complex>
#include <vector>

namespace rsatb {
namespace calc {

using cd = std::complex<double>;

inline void fft_1d(int n, cd *in, cd *out){
    fftw_plan p = fftw_plan_dft_1d(
        n, reinterpret_cast<fftw_complex*>(in), 
        reinterpret_cast<fftw_complex*>(out), 
        FFTW_FORWARD, FFTW_ESTIMATE
        );
    fftw_execute(p);
    fftw_destroy_plan(p);
    // fftw_cleanup(); // Avoid calling cleanup repeatedly in threaded env
}

inline void ifft_1d(int n, cd *in, cd *out){
    fftw_plan p = fftw_plan_dft_1d(
        n, reinterpret_cast<fftw_complex*>(in), 
        reinterpret_cast<fftw_complex*>(out), 
        FFTW_BACKWARD, FFTW_ESTIMATE
        );

    fftw_execute(p);
    fftw_destroy_plan(p);
}

inline void bifft_1d(int n, int nb, cd* in, cd* out) {
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
}

inline void bfft_1d(int n, int nb, cd* in, cd* out) {
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
}

class RowFFTWorkspace {
public:
    explicit RowFFTWorkspace(int n, int batch = 1)
        : n_(n),
          batch_(std::max(1, batch)),
          fft_n_(2 * n),
          fft_in_(static_cast<size_t>(batch_) * static_cast<size_t>(fft_n_)),
          fft_out_(static_cast<size_t>(batch_) * static_cast<size_t>(fft_n_)) {
        int fft_size[1] = {fft_n_};
        ifft_plan_ = fftw_plan_dft_1d(
            fft_n_,
            reinterpret_cast<fftw_complex*>(fft_in_.data()),
            reinterpret_cast<fftw_complex*>(fft_out_.data()),
            FFTW_BACKWARD,
            FFTW_ESTIMATE
        );
        if (batch_ > 1) {
            fftw_destroy_plan(ifft_plan_);
            ifft_plan_ = fftw_plan_many_dft(
                1,
                fft_size,
                batch_,
                reinterpret_cast<fftw_complex*>(fft_in_.data()),
                nullptr,
                1,
                fft_n_,
                reinterpret_cast<fftw_complex*>(fft_out_.data()),
                nullptr,
                1,
                fft_n_,
                FFTW_BACKWARD,
                FFTW_ESTIMATE
            );
        }
    }

    ~RowFFTWorkspace() {
        if (ifft_plan_) {
            fftw_destroy_plan(ifft_plan_);
        }
    }

    int batch_size() const {
        return batch_;
    }

    void dct_type3_batch(const cd* in, cd* out) {
        // Build an even extension with the half-grid phase shift so the
        // inverse FFT produces
        //   y_k = x_0 + 2 sum_{m=1}^{n-1} x_m cos(pi m (k + 1/2) / n)
        // for complex-valued x_m. This matches the FFTCheb cosine transform
        // used for phi^3 in the paper and avoids the brittle r2r plan path.
        std::fill(fft_in_.begin(), fft_in_.end(), cd(0.0, 0.0));
        for (int batch_idx = 0; batch_idx < batch_; ++batch_idx) {
            const size_t base = static_cast<size_t>(batch_idx) * static_cast<size_t>(n_);
            const size_t fft_base = static_cast<size_t>(batch_idx) * static_cast<size_t>(fft_n_);
            fft_in_[fft_base] = in[base];
            for (int m = 1; m < n_; ++m) {
                const double theta = M_PI * static_cast<double>(m) / (2.0 * static_cast<double>(n_));
                const cd phase_pos = std::exp(cd(0.0, theta));
                const cd phase_neg = std::exp(cd(0.0, -theta));
                const cd value = in[base + static_cast<size_t>(m)];
                fft_in_[fft_base + static_cast<size_t>(m)] = value * phase_pos;
                fft_in_[fft_base + static_cast<size_t>(fft_n_ - m)] = value * phase_neg;
            }
        }

        fftw_execute(ifft_plan_);
        for (int batch_idx = 0; batch_idx < batch_; ++batch_idx) {
            const size_t base = static_cast<size_t>(batch_idx) * static_cast<size_t>(n_);
            const size_t fft_base = static_cast<size_t>(batch_idx) * static_cast<size_t>(fft_n_);
            for (int i = 0; i < n_; ++i) {
                out[base + static_cast<size_t>(i)] = fft_out_[fft_base + static_cast<size_t>(i)];
            }
        }
    }

    void zero_padded_ifft_batch(const cd* in, int non_zero_count, cd* out, int copy_count) {
        std::fill(fft_in_.begin(), fft_in_.end(), cd(0.0, 0.0));
        for (int batch_idx = 0; batch_idx < batch_; ++batch_idx) {
            const size_t in_base = static_cast<size_t>(batch_idx) * static_cast<size_t>(n_);
            const size_t fft_base = static_cast<size_t>(batch_idx) * static_cast<size_t>(fft_n_);
            for (int i = 0; i < non_zero_count; ++i) {
                fft_in_[fft_base + static_cast<size_t>(i)] = in[in_base + static_cast<size_t>(i)];
            }
        }

        fftw_execute(ifft_plan_);
        for (int batch_idx = 0; batch_idx < batch_; ++batch_idx) {
            const size_t out_base = static_cast<size_t>(batch_idx) * static_cast<size_t>(n_);
            const size_t fft_base = static_cast<size_t>(batch_idx) * static_cast<size_t>(fft_n_);
            for (int i = 0; i < copy_count; ++i) {
                out[out_base + static_cast<size_t>(i)] = fft_out_[fft_base + static_cast<size_t>(i)];
            }
        }
    }

    void dct_type3(const cd* in, cd* out) {
        assert(batch_ == 1);
        dct_type3_batch(in, out);
    }

    void zero_padded_ifft(const cd* in, int non_zero_count, cd* out, int copy_count) {
        assert(batch_ == 1);
        zero_padded_ifft_batch(in, non_zero_count, out, copy_count);
    }

private:
    int n_;
    int batch_;
    int fft_n_;
    std::vector<cd> fft_in_;
    std::vector<cd> fft_out_;
    fftw_plan ifft_plan_ = nullptr;
};

} // namespace calc
} // namespace rsatb

#endif

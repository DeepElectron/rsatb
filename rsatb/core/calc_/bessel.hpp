#ifndef BESSEL_FUNCTION
#define BESSEL_FUNCTION

#include <bessel-library/bessel-library.hpp>
#include <list>

inline void build_bessel_vector(
    const double& time_step,
    const double& precision,
    double **bes_vec,
    int *n_bes
    )
{
    // Evaluate Bessel terms up to precision
    // The values may oscillates significantly for large time step, so we need to
    // check both n and n+1 to avoid false convergence.
    std::list<double> bes_list;
    bool converged = false;
    constexpr int n_max = 65536;
    for (int n = 0; n < n_max; ++n) {
        double bes_val = bessel::cyl_j(n, time_step);
        if (std::abs(bes_val) > precision) {
            bes_list.push_back(bes_val);
        } else {
            double bes_val_next = bessel::cyl_j(n + 1, time_step);
            if (std::abs(bes_val_next) > precision) {
                bes_list.push_back(bes_val);
            } else {
                converged = true;
                break;
            }
        }
    }

    // Check for non-convergence
    if (!converged) {
        std::cout << "ERROR: Bessel series not converged\n";
        std::exit(-1);
    }

    *n_bes = bes_list.size();
    *bes_vec = new double[*n_bes];
    int ptr=0;
    for (const auto& f : bes_list) {
        (*bes_vec)[ptr] = f;
        ++ptr;
    }
}

#endif
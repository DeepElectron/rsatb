// This file contains the sample method used to genereate the random / quasi random samples

#include <mkl.h>
#include <iostream>
#include <vector>

std::vector<double> normal(const int n, const int seed=777) {
    std::vector<double> samples(n);

    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, seed); // Mersenne Twister

    double mean = 0.0, stddev = 1.0;
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, n, samples.data(), mean, stddev);
    
    vslDeleteStream(&stream);

    return samples;
}

int main() {
    int n=1000;
    std::vector<double> ss(n);
    ss = normal(n);

    // Check few values
    for (int i = 0; i < n; ++i)
        std::cout << ss[i] << std::endl;
}

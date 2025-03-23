#ifndef FFTW_COMPLEX_TO_REAL_H
#define FFTW_COMPLEX_TO_REAL_H
#include "ComplexToReal.h"
#include <vector>
#include <complex>
#include <fftw3.h>

// Forward declaration of the template
template <typename T>
class FFTWComplexToReal;

// Specialization for float
template <>
class FFTWComplexToReal<float> : public ComplexToReal<float> {
public:
    explicit FFTWComplexToReal(size_t size);
    ~FFTWComplexToReal() override;

    void process(std::complex<float>* input, float* output) const override;
    std::unique_ptr<ComplexToReal<float>> clone() const override;

private:
    size_t size_;
    fftwf_plan plan_;
};
#endif

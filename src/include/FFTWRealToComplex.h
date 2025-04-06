#ifndef FFTW_REAL_TO_COMPLEX_H
#define FFTW_REAL_TO_COMPLEX_H
#include "RealToComplex.h"
#include <vector>
#include <complex>
#include <fftw3.h>

template <typename T>
class FFTWRealToComplex;

template <>
class FFTWRealToComplex<float> : public RealToComplex<float> {
public:
    explicit FFTWRealToComplex(size_t size);
    ~FFTWRealToComplex() override;

    void process(float* input, std::complex<float>* output) const override;
    std::unique_ptr<RealToComplex<float>> clone() const override;

private:
    size_t size_;
    fftwf_plan plan_;
};
#endif

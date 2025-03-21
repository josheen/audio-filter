#ifndef FFTW_REAL_TO_COMPLEX_H
#define FFTW_REAL_TO_COMPLEX_H
#include "RealToComplex.h"
#include <vector>
#include <complex>
#include <fftw3.h>

// Forward declaration of the template (not defined generically)
template <typename T>
class FFTWRealToComplex;

// Specialization for float
template <>
class FFTWRealToComplex<float> : public RealToComplex<float> {
public:
    explicit FFTWRealToComplex(size_t size);
    ~FFTWRealToComplex();

    void process(const std::vector<float>& input, std::vector<std::complex<float>>& output) const override;

    std::unique_ptr<RealToComplex<float>> clone() const override {
        return std::make_unique<FFTWRealToComplex<float>>(*this);
    }

private:
    size_t size_;
    fftwf_plan plan_;
    mutable std::vector<float> input_buffer_;
    mutable std::vector<std::complex<float>> output_buffer_;
};

// Specialization for double
template <>
class FFTWRealToComplex<double> : public RealToComplex<double> {
public:
    explicit FFTWRealToComplex(size_t size);
    ~FFTWRealToComplex();

    void process(const std::vector<double>& input, std::vector<std::complex<double>>& output) const override;
    
    std::unique_ptr<RealToComplex<double>> clone() const override {
        return std::make_unique<FFTWRealToComplex<double>>(*this);
    }

private:
    size_t size_;
    fftw_plan plan_;
    mutable std::vector<double> input_buffer_;
    mutable std::vector<std::complex<double>> output_buffer_;
};
#endif

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
    ~FFTWComplexToReal();

    void process(const std::vector<std::complex<float>>& input, std::vector<float>& output) const override;

    std::unique_ptr<ComplexToReal<float>> clone() const override {
        return std::make_unique<FFTWComplexToReal<float>>(*this);
    }

private:
    size_t size_;
    fftwf_plan plan_;
    mutable std::vector<std::complex<float>> input_buffer_;
    mutable std::vector<float> output_buffer_;
};

// Specialization for double
template <>
class FFTWComplexToReal<double> : public ComplexToReal<double> {
public:
    explicit FFTWComplexToReal(size_t size);
    ~FFTWComplexToReal();

    void process(const std::vector<std::complex<double>>& input, std::vector<double>& output) const override;
    std::unique_ptr<ComplexToReal<double>> clone() const override {
        return std::make_unique<FFTWComplexToReal<double>>(*this);
    }

private:
    size_t size_;
    fftw_plan plan_;
    mutable std::vector<std::complex<double>> input_buffer_;
    mutable std::vector<double> output_buffer_;
};
#endif


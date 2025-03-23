#include "FFTWComplexToReal.h"

FFTWComplexToReal<float>::FFTWComplexToReal(size_t size)
    : size_(size) {
    // Allocate temporary buffers for planning
    std::vector<std::complex<float>> input_buffer(size / 2 + 1);
    std::vector<float> output_buffer(size);

    plan_ = fftwf_plan_dft_c2r_1d(
        static_cast<int>(size_),
        reinterpret_cast<fftwf_complex*>(input_buffer.data()),
        output_buffer.data(),
        FFTW_MEASURE);
}

FFTWComplexToReal<float>::~FFTWComplexToReal() {
    fftwf_destroy_plan(plan_);
}

void FFTWComplexToReal<float>::process(std::complex<float>* input, float* output) const {
    fftwf_execute_dft_c2r(
        plan_,
        reinterpret_cast<fftwf_complex*>(const_cast<std::complex<float>*>(input)),
        output);
}

std::unique_ptr<ComplexToReal<float>> FFTWComplexToReal<float>::clone() const {
    return std::make_unique<FFTWComplexToReal<float>>(*this);
}

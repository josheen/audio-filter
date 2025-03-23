#include "FFTWRealToComplex.h"

FFTWRealToComplex<float>::FFTWRealToComplex(size_t size)
    : size_(size) {
    // Allocate temporary buffers for planning
    std::vector<float> input_buffer(size);
    std::vector<std::complex<float>> output_buffer(size / 2 + 1);

    plan_ = fftwf_plan_dft_r2c_1d(
        static_cast<int>(size_), input_buffer.data(),
        reinterpret_cast<fftwf_complex*>(output_buffer.data()),
        FFTW_MEASURE);
}

FFTWRealToComplex<float>::~FFTWRealToComplex() {
    fftwf_destroy_plan(plan_);
}

void FFTWRealToComplex<float>::process(float* input, std::complex<float>* output) const {
    fftwf_execute_dft_r2c(plan_, input, reinterpret_cast<fftwf_complex*>(output));
}

std::unique_ptr<RealToComplex<float>> FFTWRealToComplex<float>::clone() const {
    return std::make_unique<FFTWRealToComplex<float>>(*this);
}

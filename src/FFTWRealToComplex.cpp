#include "FFTWRealToComplex.h"

// Float specialization
FFTWRealToComplex<float>::FFTWRealToComplex(size_t size)
    : size_(size),
      input_buffer_(size),
      output_buffer_(size / 2 + 1)
{
    plan_ = fftwf_plan_dft_r2c_1d(
        static_cast<int>(size_),
        input_buffer_.data(),
        reinterpret_cast<fftwf_complex*>(output_buffer_.data()),
        FFTW_MEASURE
    );
}

FFTWRealToComplex<float>::~FFTWRealToComplex() {
    fftwf_destroy_plan(plan_);
}

void FFTWRealToComplex<float>::process(const std::vector<float>& input, std::vector<std::complex<float>>& output) const {
    input_buffer_ = input;
    fftwf_execute_dft_r2c(plan_,
                          input_buffer_.data(),
                          reinterpret_cast<fftwf_complex*>(output_buffer_.data()));
    output = output_buffer_;
}

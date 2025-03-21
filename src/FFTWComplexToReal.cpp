#include "FFTWComplexToReal.h"

// Float specialization
FFTWComplexToReal<float>::FFTWComplexToReal(size_t size)
    : size_(size),
      input_buffer_(size / 2 + 1),
      output_buffer_(size)
{
    plan_ = fftwf_plan_dft_c2r_1d(
        static_cast<int>(size_),
        reinterpret_cast<fftwf_complex*>(input_buffer_.data()),
        output_buffer_.data(),
        FFTW_MEASURE
    );
}

FFTWComplexToReal<float>::~FFTWComplexToReal() {
    fftwf_destroy_plan(plan_);
}

void FFTWComplexToReal<float>::process(const std::vector<std::complex<float>>& input, std::vector<float>& output) const {
    input_buffer_ = input;
    fftwf_execute_dft_c2r(plan_,
                          reinterpret_cast<fftwf_complex*>(input_buffer_.data()),
                          output_buffer_.data());
    output = output_buffer_;
}

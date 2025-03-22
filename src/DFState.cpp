#include <cmath>
#include <cassert>
#include "DFState.h"
#include "RealFFTPlanner.h"
#include "ERB_pub.h"
#include "FFTWRealToComplex.h"
#include "FFTWComplexToReal.h"

DFState::DFState(size_t sr, size_t fft_size, size_t hop_size, size_t nb_bands, size_t min_nb_freqs)
    : sr_(sr),
      frame_size_(hop_size),
      window_size_(fft_size),
      freq_size_(fft_size / 2 + 1),
      wnorm_(1.0f / (std::pow(fft_size, 2) / static_cast<float>(2 * hop_size))),
      analysis_mem_(fft_size - hop_size, 0.0f),
      synthesis_mem_(fft_size - hop_size, 0.0f),
      window_(fft_size, 0.0f)
{
    assert(hop_size * 2 <= fft_size);

    RealFFTPlanner<float> fft(
            [](size_t size) -> std::shared_ptr<RealToComplex<float>> {
            return std::make_shared<FFTWRealToComplex<float>>(size);
            },
            [](size_t size) -> std::shared_ptr<ComplexToReal<float>> {
            return std::make_shared<FFTWComplexToReal<float>>(size);
            }
            );
    fft_forward_ = fft.plan_fft_forward(fft_size);
    fft_inverse_ = fft.plan_fft_inverse(fft_size);

    erb_ = erb_fb(sr, fft_size, nb_bands, min_nb_freqs);
    constexpr double pi = M_PI;
    size_t window_size_h = fft_size / 2;
    for (size_t i = 0; i < fft_size; ++i) {
        double s = sin(0.5 * pi * (static_cast<double>(i) + 0.5) / static_cast<double>(window_size_h));
        window_[i] = static_cast<float>(sin(0.5 * pi * s * s));
    }

    mean_norm_state_.clear();
    unit_norm_state_.clear();
}


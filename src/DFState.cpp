#include <cmath>
#include <cassert>
#include "DFState.h"
#include "ERB_pub.h"
#include "RealFFTPlanner.h"
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
            });
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

constexpr float MEAN_NORM_INIT_MIN = -60.0f;
constexpr float MEAN_NORM_INIT_MAX = -90.0f;
constexpr float UNIT_NORM_INIT_MIN = 0.001f;
constexpr float UNIT_NORM_INIT_MAX = 0.0001f;

void DFState::init_norm_states(size_t nb_df_freqs) {
    DFState::init_mean_norm_state();
    DFState::init_unit_norm_state(nb_df_freqs);
}

void DFState::init_mean_norm_state() {
    size_t nb_erb = erb_.size();
    float step = (MEAN_NORM_INIT_MAX - MEAN_NORM_INIT_MIN) / static_cast<float>(nb_erb - 1);

    mean_norm_state_.clear();
    mean_norm_state_.reserve(nb_erb);
    for (size_t i = 0; i < nb_erb; i++) {
        mean_norm_state_.push_back(MEAN_NORM_INIT_MIN + static_cast<float>(i) * step);
    }
}

void DFState::init_unit_norm_state(size_t nb_freqs) {
    float step = (UNIT_NORM_INIT_MAX - UNIT_NORM_INIT_MIN) / static_cast<float>(nb_freqs - 1);

    unit_norm_state_.clear();
    unit_norm_state_.reserve(nb_freqs);
    for (size_t i = 0; i < nb_freqs; i++) {
        unit_norm_state_.push_back(UNIT_NORM_INIT_MIN + static_cast<float>(i) * step);
    }
}


std::vector<size_t> erb_fb(size_t sr, size_t fft_size,size_t nb_bands, size_t min_nb_freqs) {
    size_t nyq_freq = sr / 2;
    float freq_width = static_cast<float>(sr) / static_cast<float>(fft_size);
    float erb_low = freq2erb(0.0f);
    float erb_high = freq2erb(static_cast<float>(nyq_freq));
    std::vector<size_t> erb(nb_bands, 0);
    float step = (erb_high - erb_low) / static_cast<float>(nb_bands);
    int prev_freq = 0;
    int freq_over = 0;

    for (int i = 0; i <= min_nb_freqs; i++) {
        float f = erb2freq(erb_low + static_cast<float>(i) * step);
        size_t fb = static_cast<size_t>(std::round(f / freq_width));
        int nb_freqs = static_cast<int>(fb) - prev_freq - freq_over;
        if (nb_freqs < static_cast<int>(min_nb_freqs)) {
            freq_over = static_cast<int>(min_nb_freqs) - nb_freqs;
            nb_freqs = static_cast<int>(min_nb_freqs);
        } else {
            freq_over = 0;
        }
        erb[i - 1] = static_cast<size_t>(nb_freqs);
        prev_freq = fb;
    }
    erb[nb_bands - 1] += 1;
    size_t too_large = 0;
    size_t sum = 0;
    for (const auto val : erb) sum += val;
    too_large = sum - (fft_size / 2 + 1);
    if (too_large > 0) {
        erb[nb_bands - 1] -= too_large;
    }
    return erb;
}

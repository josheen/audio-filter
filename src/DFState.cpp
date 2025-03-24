#include <cmath>
#include <cassert>
#include "DFState.h"
#include "ERB_pub.h"
#include "RealFFTPlanner.h"
#include "FFTWRealToComplex.h"
#include "FFTWComplexToReal.h"
#include <numeric>

void compute_band_corr(Eigen::Map<Eigen::ArrayXf>& out, const Eigen::Map<const Eigen::ArrayXcf>& x,
        const Eigen::Map<const Eigen::ArrayXcf>& p, const std::vector<size_t>& erb_fb);
void band_mean_norm_erb(Eigen::Map<Eigen::ArrayXf>& xs, std::vector<float>& state, float alpha);
void band_unit_norm(Eigen::Map<Eigen::ArrayXcf>& xs, std::vector<float>& state, float alpha);

void DFState::feat_erb(const Eigen::Map<const Eigen::ArrayXcf>& input, float alpha, Eigen::Map<Eigen::ArrayXf>& output) {
    compute_band_corr(output, input, input, erb_);
    for (auto& o : output) {
        o = 10.0f * std::log10(o + 1e-10f);
    }
    band_mean_norm_erb(output, mean_norm_state_, alpha);
}

void DFState::feat_cplx(const Eigen::Ref<const Eigen::ArrayXcf>& input, float alpha, Eigen::Map<Eigen::ArrayXcf>& output) {
    output = input;
    band_unit_norm(output, unit_norm_state_, alpha);
}

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

void DFState::analysis(const float* input, std::complex<float>* output) {
    assert(input != nullptr);
    assert(output != nullptr);
    assert(frame_size_ > 0 && freq_size_ > 0);

    frame_analysis(input, output, *this);
}

void DFState::frame_analysis(const float* input, std::complex<float>* output, DFState& state) {
    assert(state.frame_size_ > 0 && state.freq_size_ > 0);

    std::vector<float> buf(state.window_size_, 0.0f);

    size_t split = state.window_size_ - state.frame_size_;
    auto buf_first = buf.begin();
    auto buf_second = buf.begin() + split;
    auto window_first = state.window_.begin();
    auto window_second = state.window_.begin() + split;

    // Fill first part of buf from analysis_mem_ * window_first
    for (size_t i = 0; i < split; i++) {
        buf_first[i] = state.analysis_mem_[i] * window_first[i];
    }

    // Fill second part from input * window_second
    for (size_t i = 0; i < state.frame_size_; i++) {
        buf_second[i] = input[i] * window_second[i];
    }

    // Rotate analysis_mem_ if hop size < window_size / 2
    size_t analysis_split = state.analysis_mem_.size() - state.frame_size_;
    if (analysis_split > 0) {
        std::rotate(state.analysis_mem_.begin(),
                    state.analysis_mem_.begin() + state.frame_size_,
                    state.analysis_mem_.end());
    }

    for (size_t i = 0; i < state.frame_size_; i++) {
        state.analysis_mem_[analysis_split + i] = input[i];
    }

    state.fft_forward_->process(buf.data(), output);

    // Normalize
    for (size_t i = 0; i < state.freq_size_; i++) {
        output[i] *= state.wnorm_;
    }
}

void DFState::synthesis(std::complex<float>* input, float* output) {
    std::vector<float> x(window_size_, 0.0f);

    fft_inverse_->process(input, x.data());

    // Apply synthesis window
    for (size_t i = 0; i < window_size_; i++) {
        x[i] *= window_[i];
    }

    // Split the inverse FFT output
    auto x_first = x.begin();
    auto x_second = x.begin() + frame_size_;

    // Overlap-add: Add synthesis memory with first part of output
    for (size_t i = 0; i < frame_size_; i++) {
        output[i] = x_first[i] + synthesis_mem_[i];
    }

    // Rotate synthesis memory if overlap (hop < window_size_/2)
    size_t split = synthesis_mem_.size() - frame_size_;
    if (split > 0) {
        std::rotate(synthesis_mem_.begin(), synthesis_mem_.begin() + frame_size_, synthesis_mem_.end());
    }

    // Update synthesis_mem_ with the second part of FFT output
    auto s_first = synthesis_mem_.begin();
    auto s_second = synthesis_mem_.begin() + split;
    auto xs_first = x_second;
    auto xs_second = x_second + split;

    // Add overlap part
    for (size_t i = 0; i < split; i++) {
        s_first[i] += xs_first[i];
    }

    // Replace memory part
    for (size_t i = 0; i < (window_size_ - frame_size_ - split); i++) {
        s_second[i] = xs_second[i];
    }
}


void DFState::apply_mask(Eigen::Ref<Eigen::ArrayXcf> output, const Eigen::ArrayXf& gains) {
    apply_interp_band_gain(output, gains, erb_);
}
void DFState::apply_interp_band_gain(Eigen::Ref<Eigen::ArrayXcf>& out, const Eigen::ArrayXf& band_gains, const std::vector<size_t>& erb_fb) {
    size_t offset = 0;
    for (size_t i = 0; i < erb_fb.size(); i++) {
        float gain = band_gains(i);
        size_t band_size = erb_fb[i];
        for (size_t j = 0; j < band_size; j++) {
            out(offset + j) *= gain;
        }
        offset += band_size;
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

    for (int i = 1; i <= min_nb_freqs; i++) {
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
    size_t sum = std::accumulate(erb.begin(), erb.end(),0U);
    too_large = sum - (fft_size / 2 + 1);
    if (too_large > 0) {
        erb[nb_bands - 1] -= too_large;
    }
    return erb;
}


void compute_band_corr(Eigen::Map<Eigen::ArrayXf>& out, const Eigen::Map<const Eigen::ArrayXcf>& x,
        const Eigen::Map<const Eigen::ArrayXcf>& p, const std::vector<size_t>& erb_fb) {
    std::fill(out.begin(), out.end(), 0.0f);

    // Ensure erb_fb and out have the same size
    assert(erb_fb.size() == out.size());

    size_t bcsum = 0; // Cumulative index for bands
    for (size_t i = 0; i < erb_fb.size(); ++i) {
        size_t band_size = erb_fb[i]; // Size of the current band
        float k = 1.0f / static_cast<float>(band_size); // Normalization factor

        // Compute the correlation for the current band
        for (size_t j = 0; j < band_size; ++j) {
            size_t idx = bcsum + j; // Index into x and p
            out[i] += (x[idx].real() * p[idx].real() + x[idx].imag() * p[idx].imag()) * k;
        }

        bcsum += band_size; // Update cumulative index
    }
}
void band_mean_norm_erb(Eigen::Map<Eigen::ArrayXf>& xs, std::vector<float>& state, float alpha) {
    // Ensure xs and state have the same size
    assert(xs.size() == state.size());

    // Apply exponential mean normalization
    for (size_t i = 0; i < xs.size(); ++i) {
        // Update the state with the exponential moving average
        state[i] = xs[i] * (1.0f - alpha) + state[i] * alpha;

        // Normalize the input value
        xs[i] -= state[i];
        xs[i] /= 40.0f;
    }
}


void band_unit_norm(Eigen::Map<Eigen::ArrayXcf>& xs, std::vector<float>& state, float alpha) {
    assert(xs.size() == state.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        state[i] = std::abs(xs[i]) * (1.0f - alpha) + state[i] * alpha;
        xs[i] /= std::sqrt(state[i]);
    }
}

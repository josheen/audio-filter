#ifndef DF_STATE_H
#define DF_STATE_H

#include <vector>
#include <complex>
#include "RealToComplex.h"
#include <Eigen/Dense>
#include "ComplexToReal.h"

template <typename T>
class ArrayViewMut2 {
public:
    ArrayViewMut2(T* data, size_t rows, size_t cols)
        : data_(data), rows_(rows), cols_(cols) {}

    T& operator()(size_t row, size_t col) {
        return data_[row * cols_ + col];
    }

    const T& operator()(size_t row, size_t col) const {
        return data_[row * cols_ + col];
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

private:
    T* data_;
    size_t rows_;
    size_t cols_;
};


class DFState {
    public:
        DFState(size_t sr, size_t fft_size, size_t hop_size, size_t nb_bands, size_t min_nb_freqs);
        virtual ~DFState() = default;
        void init_norm_states(size_t nb_df_freqs);
        void init_mean_norm_state();
        void init_unit_norm_state(size_t nb_freqs);
        void analysis(const float* input, std::complex<float>* output);
        void feat_erb(const Eigen::Map<const Eigen::ArrayXcf>& input, float alpha, Eigen::Map<Eigen::ArrayXf>& output);
        void feat_cplx(const Eigen::Ref<const Eigen::ArrayXcf>& input, float alpha, Eigen::Map<Eigen::ArrayXcf>& output);
        void frame_analysis(const float* input, std::complex<float>* output, DFState& state);
        void synthesis(std::complex<float>* input, float* output);
        void apply_mask(Eigen::Ref<Eigen::ArrayXcf> spec, const Eigen::ArrayXf& gains);
        std::vector<float> mean_norm_state_;
        std::vector<float> unit_norm_state_;

    private:
        void apply_interp_band_gain(Eigen::Ref<Eigen::ArrayXcf>& out, const Eigen::ArrayXf& band_e, const std::vector<size_t>& erb_fb);
        size_t sr_;
        size_t frame_size_;
        size_t window_size_;
        size_t freq_size_;

        std::shared_ptr<RealToComplex<float>> fft_forward_;
        std::shared_ptr<ComplexToReal<float>> fft_inverse_;

        std::vector<float> window_;
        float wnorm_;
        std::vector<size_t> erb_;
        std::vector<float> analysis_mem_;
        std::vector<float> synthesis_mem_;
};
#endif

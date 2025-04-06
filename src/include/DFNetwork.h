#ifndef DF_NETWORK_H
#define DF_NETWORK_H

#include "DFParams.h"
#include "DFState.h"
#include "RuntimeParams.h"
#include <deque>
#include <onnxruntime_cxx_api.h>
#include <Eigen/Dense>
#include <Eigen/Core>

struct TensorBuffer {
    std::vector<float> data;
    std::vector<int64_t> shape;
    TensorBuffer() = default;
    // Copy constructor
    TensorBuffer(const TensorBuffer& other)
        : data(other.data), shape(other.shape) {}
};

struct TensorComplex {
    std::vector<std::complex<float>> data;
    std::vector<int64_t> shape;
    TensorComplex() = default;
    // Copy constructor
    TensorComplex(const TensorComplex& other)
        : data(other.data), shape(other.shape) {}
};

class DFNetwork {
    public:
        DFNetwork(const DFParams& df_params, const RuntimeParams& rp, Ort::Env& env, Ort::SessionOptions& session_options);
        float process(const Eigen::MatrixXf& noisy_frame, Eigen::MatrixXf& enhanced_frame);
        std::tuple<float, std::optional<TensorBuffer>, std::optional<TensorBuffer>> process_raw();
        std::tuple<bool, bool, bool> apply_stages(float lsnr) const;
        void init();
        std::deque<TensorComplex> rolling_spec_buf_y_;
        std::deque<TensorComplex> rolling_spec_buf_x_;
        size_t ch_;
        size_t df_order_;
        size_t conv_lookahead_;
        size_t n_freqs_;
        TensorComplex spec_buf_;
        Ort::MemoryInfo mem_info_;
        Ort::Session enc_session_;
        Ort::Session erb_dec_session_;
        Ort::Session df_dec_session_;
        // Buffers:
        std::deque<TensorBuffer> rolling_erb;  // Store past erb feature frames
        std::deque<TensorBuffer> rolling_cplx; // Store past cplx feature frames
        std::deque<std::vector<float>> rolling_c0;     // Store old C0 tensors
        TensorBuffer erb_buf_;
        TensorComplex cplx_buf_;
        std::vector<float> m_zeros_;
        size_t sr_;
        size_t fft_size_;
        size_t hop_size_;
        size_t nb_erb_;
        size_t nb_df_;
        size_t lookahead_;
        std::vector<DFState> df_states_;
        TensorBuffer enc_hidden;
        TensorBuffer erb_dec_hidden;
        TensorBuffer df_dec_hidden;
    private:
        float calc_norm_alpha(size_t sr, size_t hop_size, float tau) {
            float dt = static_cast<float>(hop_size) / static_cast<float>(sr);
            float alpha = std::exp(-dt / tau);
            float a = 1.0f;
            int precision = 3;
            while (a >= 1.0f) {
                float factor = std::pow(10.0f, static_cast<float>(precision));
                a = (alpha * factor) / factor;
                precision += 1;
            }
            return a;
        }
        void df(
                const std::deque<TensorComplex>& spec_frames,
                const TensorBuffer& coefs,
                size_t nb_df,
                size_t df_order,
                size_t n_freqs,
                TensorComplex& spec_out
               );
        size_t df_lookahead;
        size_t min_nb_erb_freqs_;
        float alpha_;
        float min_db_thresh_;
        float max_db_erb_thresh_;
        float max_db_df_thresh_;
        ReduceMask reduce_mask_;
        std::optional<float> atten_lim_;
        bool post_filter_;
        float post_filter_beta_;
        size_t skip_counter_;
        // Encoder I/O names
        const std::array<const char*, 3> encoder_input_names_ = {"feat_erb", "feat_spec", "enc_hidden"};
        const std::array<const char*, 8> encoder_output_names_ = {"e0", "e1", "e2", "e3", "emb", "c0", "lsnr", "hidden"};
        // ERB decoder I/O names
        const std::array<const char*, 6> erb_input_names_ = {"emb", "e3", "e2", "e1", "e0", "erb_dec_hidden"};
        const std::array<const char*, 2> erb_dec_output_names_ = {"m", "hidden"};
        // DF decoder I/O names
        const std::array<const char*, 3> df_dec_input_names_ = {"emb", "c0", "df_dec_hidden"};
        const std::array<const char*, 2> df_dec_output_names_ = {"coefs", "hidden"};
};

#endif

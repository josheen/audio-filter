#ifndef DF_NETWORK_H
#define DF_NETWORK_H

#include "DFParams.h"
#include "DFState.h"
#include "RuntimeParams.h"
#include <deque>
#include <onnxruntime_cxx_api.h>
#include <Eigen/Dense>

struct TensorBuffer {
    std::vector<float> data;
    std::vector<int64_t> shape;

    Ort::Value to_ort_tensor(Ort::MemoryInfo& mem_info) {
        return Ort::Value::CreateTensor<float>(
            mem_info,
            data.data(),
            data.size(),
            shape.data(),
            shape.size()
        );
    }
};

class DFNetwork {
    public:
        DFNetwork(const DFParams& df_params, const RuntimeParams& rp, Ort::Env& env, Ort::SessionOptions& session_options);
        float process(const Eigen::MatrixXf& noisy_frame, Eigen::MatrixXf& enhanced_frame);
        void init();
        std::deque<std::vector<float>> rolling_spec_buf_y_;
        std::deque<std::vector<float>> rolling_spec_buf_x_;
        size_t ch_;
        size_t df_order_;
        size_t conv_lookahead_;
        size_t n_freqs_;
        TensorBuffer spec_buf_;
        Ort::MemoryInfo mem_info_;
        Ort::Session enc_session_;
        Ort::Session erb_dec_session_;
        Ort::Session df_dec_session_;
        // Buffers:
        TensorBuffer erb_buf_;
        TensorBuffer cplx_buf_;
        std::vector<float> m_zeros_;
        size_t sr_;
        size_t fft_size_;
        size_t hop_size_;
        size_t nb_erb_;
        size_t nb_df_;
        size_t lookahead_;
        std::vector<DFState> df_states_;
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
        size_t df_lookahead;
        size_t min_nb_erb_freqs_;
        float alpha_;
        float min_db_thresh;
        float max_db_erb_thresh;
        float max_db_df_thresh;
        ReduceMask reduce_mask_;
        std::optional<float> atten_lim_;
        bool post_filter_;
        float post_filter_beta_;
        size_t skip_counter_;
};

#endif

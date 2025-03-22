#include "DFParams.h"
#include "DFState.h"
#include "RuntimeParams.h"
#include <deque>

// DFNetwork class definition
class DFNetwork {
    public:
        DFNetwork(const DFParams& df_params, const RuntimeParams& rp, Ort::Env& env, Ort::SessionOptions& session_options);
        void process_audio(const std::vector<float>& audio_input, std::vector<float>& audio_output);
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
        Ort::Session enc_session_;
        Ort::Session erb_dec_session_;
        Ort::Session df_dec_session_;

        size_t lookahead_;
        size_t df_lookahead;
        size_t conv_lookahead_;
        size_t sr_;
        size_t ch_;
        size_t fft_size_;
        size_t hop_size_;
        size_t nb_erb_;
        size_t min_nb_erb_freqs_;
        size_t nb_df_;
        size_t n_freqs_;
        size_t df_order_;
        float alpha_;
        float min_db_thresh;
        float max_db_erb_thresh;
        float max_db_df_thresh;
        ReduceMask reduce_mask_;
        std::optional<float> atten_lim_;

        std::vector<DFState> df_states_;

        // Buffers:
        std::vector<float> spec_buf_;
        std::vector<float> erb_buf_;
        std::vector<float> cplx_buf_;
        std::vector<float> m_zeros_;
        std::deque<std::vector<float>> rolling_spec_buf_y_;
        std::deque<std::vector<float>> rolling_spec_buf_x_;

        bool post_filter_;
        float post_filter_beta_;
        size_t skip_counter_;
};

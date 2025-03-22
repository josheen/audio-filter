#include <map>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include "DFParams.h"
#include "DFState.h"
#include "RuntimeParams.h"

// DFNetwork class definition
class DFNetwork {
    public:
        DFNetwork(const DFParams& df_params, const RuntimeParams& rp, Ort::Env& env, Ort::SessionOptions& session_options)
            : enc_session_(env, df_params.enc_.data(), df_params.enc_.size(), session_options),
            erb_dec_session_(env, df_params.erb_dec_.data(), df_params.erb_dec_.size(), session_options),
            df_dec_session_(env, df_params.df_dec_.data(), df_params.df_dec_.size(), session_options) {

            auto model_cfg = df_params.section("deepfilternet");
            auto df_cfg = df_params.section("df");

            // Config parameters
            sr_ = std::stoi(df_cfg.at("sr"));
            hop_size_ = std::stoi(df_cfg.at("hop_size"));
            fft_size_ = std::stoi(df_cfg.at("fft_size"));
            min_nb_erb_freqs_ = std::stoi(df_cfg.at("min_nb_erb_freqs"));
            nb_erb_ = std::stoi(df_cfg.at("nb_erb"));
            nb_df_ = std::stoi(df_cfg.at("nb_df"));
            df_order_ = (df_cfg.find("df_order") != df_cfg.end())
                ? std::stoi(df_cfg.at("df_order"))
                : std::stoi(model_cfg.at("df_order"));
            conv_lookahead_ = std::stoi(model_cfg.at("conv_lookahead"));
            df_lookahead = (df_cfg.find("df_lookahead") != df_cfg.end())
                ? std::stoi(df_cfg.at("df_lookahead"))
                : std::stoi(model_cfg.at("df_lookahead"));
            ch_ = rp.n_ch_;
            n_freqs_ = fft_size_ / 2 + 1;

            // Calculate alpha
            if (df_cfg.find("norm_alpha") != df_cfg.end()) {
                alpha_ = std::stof(df_cfg.at("norm_alpha"));
            } else {
                float tau = std::stof(df_cfg.at("norm_tau"));
                alpha_ = calc_norm_alpha(sr_, hop_size_, tau);
            }

            // Attenuation limit logic
            float atten_lim_db = std::abs(rp.atten_lim_db_);
            if (atten_lim_db >= 100.0f) {
                atten_lim_ = std::nullopt;
            } else if (atten_lim_db < 0.01f) {
                std::cout << "Attenuation limit too strong. No noise reduction will be performed.\n";
                atten_lim_ = 1.0f;
            } else {
                std::cout << "Running with attenuation limit of " << atten_lim_db << " dB\n";
                atten_lim_ = std::pow(10.0f, -atten_lim_db / 20.0f);
            }

            spec_buf_.resize(n_freqs_ * 2, 0.0f);
            erb_buf_.resize(nb_erb_, 0.0f);
            cplx_buf_.resize(nb_df_ * 2, 0.0f);
            m_zeros_.resize(nb_erb_, 0.0f);

            // Model type detection
            std::string model_type = df_params.section("train").at("model");
            if (model_type == "deepfilternet3") {
                lookahead_ = std::max(conv_lookahead_, df_lookahead);
            } else {
                throw std::runtime_error("Unsupported model type: " + model_type);
            }
            std::cout << "Running with model type " << model_type << " and lookahead " << lookahead_ << std::endl;

            // Setup state
            DFState state(sr_, fft_size_, hop_size_, nb_erb_, min_nb_erb_freqs_);
            state.init_norm_states(nb_df_);
            df_states_.push_back(state);

            // Pre-allocate rolling buffers if needed
            rolling_spec_buf_y_ = std::deque<std::vector<float>>(df_order_ + lookahead_);
            rolling_spec_buf_x_ = std::deque<std::vector<float>>(std::max(lookahead_, df_order_));

            // Store runtime params
            post_filter_ = rp.post_filter_;
            post_filter_beta_ = rp.post_filter_beta_;
            min_db_thresh = rp.min_db_thresh_;
            max_db_erb_thresh = rp.max_db_erb_thresh_;
            max_db_df_thresh = rp.max_db_df_thresh_;
            reduce_mask_ = rp.reduce_mask_;
            skip_counter_ = 0;
        }

        // Example stub for processing (can be extended later)
        void process_audio(const std::vector<float>& audio_input, std::vector<float>& audio_output) {
            std::cout << "Processing audio with DFNetwork...\n";
            audio_output = audio_input;
        }

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

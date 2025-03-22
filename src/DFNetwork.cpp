#include "DFNetwork.h"
#include <string>
#include <stdexcept>
#include <iostream>

DFNetwork::DFNetwork(const DFParams& df_params, const RuntimeParams& rp, Ort::Env& env, Ort::SessionOptions& session_options)
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
void DFNetwork::process_audio(const std::vector<float>& audio_input, std::vector<float>& audio_output) {
    std::cout << "Processing audio with DFNetwork...\n";
    audio_output = audio_input;
}


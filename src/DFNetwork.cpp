#include "DFNetwork.h"
#include <string>
#include <stdexcept>
#include <iostream>

DFNetwork::DFNetwork(const DFParams& df_params, const RuntimeParams& rp, Ort::Env& env, Ort::SessionOptions& session_options)
    : enc_session_(env, df_params.enc_.data(), df_params.enc_.size(), session_options),
    erb_dec_session_(env, df_params.erb_dec_.data(), df_params.erb_dec_.size(), session_options),
    df_dec_session_(env, df_params.df_dec_.data(), df_params.df_dec_.size(), session_options),
    mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

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

        spec_buf_.data.resize(n_freqs_ * 2, 0.0f);
        spec_buf_.shape = {1, 1, 1, static_cast<int64_t>(n_freqs_), 2};

        erb_buf_.data.resize(ch_ * nb_erb_, 0.0f);
        erb_buf_.shape = {static_cast<int64_t>(ch_), 1, 1, static_cast<int64_t>(nb_erb_)};

        cplx_buf_.data.resize(ch_ * nb_df_ * 2, 0.0f);
        cplx_buf_.shape = {static_cast<int64_t>(ch_), 1, static_cast<int64_t>(nb_df_), 2};
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
        size_t spec_buf_size = ch_ * 1 * 1 * n_freqs_ * 2;
        rolling_spec_buf_y_.clear();
        for (size_t i = 0; i < (df_order_ + lookahead_); i++) {
            rolling_spec_buf_y_.emplace_back(std::vector<float>(spec_buf_size, 0.0f));
        }
        rolling_spec_buf_x_.clear();
        for (size_t i = 0; i < std::max(lookahead_, df_order_); i++) {
            rolling_spec_buf_x_.emplace_back(std::vector<float>(spec_buf_size, 0.0f));
        }

        // Store runtime params
        post_filter_ = rp.post_filter_;
        post_filter_beta_ = rp.post_filter_beta_;
        min_db_thresh = rp.min_db_thresh_;
        max_db_erb_thresh = rp.max_db_erb_thresh_;
        max_db_df_thresh = rp.max_db_df_thresh_;
        reduce_mask_ = rp.reduce_mask_;
        skip_counter_ = 0;
        init();
    }
void DFNetwork::init() {
    size_t spec_buf_size = ch_ * 1 * 1 * n_freqs_ * 2;
    std::cout << "Initializing DFNetwork rolling buffers and state...\n";

    // Reset rolling_spec_buf_y_ with zero-filled vectors
    rolling_spec_buf_y_.clear();
    for (size_t i = 0; i < (df_order_ + conv_lookahead_); ++i) {
        rolling_spec_buf_y_.emplace_back(std::vector<float>(spec_buf_size, 0.0f));
    }

    rolling_spec_buf_x_.clear();
    for (size_t i = 0; i < std::max(df_order_, lookahead_); ++i) {
        rolling_spec_buf_x_.emplace_back(std::vector<float>(spec_buf_size, 0.0f));
    }

    // Make sure we have a DFState per channel
    if (ch_ > df_states_.size()) {
        for (size_t i = df_states_.size(); i < ch_; ++i) {
            DFState state(sr_, fft_size_, hop_size_, nb_erb_, min_nb_erb_freqs_);
            state.init_norm_states(nb_df_);
            df_states_.push_back(state);
        }
    }

    // Reset spec_buf_, erb_buf_, cplx_buf_ with correct zero data and shape
    spec_buf_.data.assign(spec_buf_size, 0.0f);
    spec_buf_.shape = {static_cast<int64_t>(ch_), 1, 1, static_cast<int64_t>(n_freqs_), 2};

    erb_buf_.data.assign(ch_ * nb_erb_, 0.0f);
    erb_buf_.shape = {static_cast<int64_t>(ch_), 1, 1, static_cast<int64_t>(nb_erb_)};

    cplx_buf_.data.assign(ch_ * nb_df_ * 2, 0.0f);
    cplx_buf_.shape = {static_cast<int64_t>(ch_), 1, static_cast<int64_t>(nb_df_), 2};

    std::cout << "DFNetwork::init() completed.\n";
}

// Example stub for processing (can be extended later)
void DFNetwork::process_audio(const std::vector<float>& audio_input, std::vector<float>& audio_output) {
    std::cout << "Processing audio with DFNetwork...\n";
    audio_output = audio_input;
}


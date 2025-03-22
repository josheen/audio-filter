#include "df_utils.h"
#include <map>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cstdint>

RuntimeParams::RuntimeParams(
    size_t n_ch,
    float post_filter_beta,
    float atten_lim_db,
    float min_db_thresh,
    float max_db_erb_thresh,
    float max_db_df_thresh,
    ReduceMask reduce_mask
)
    : n_ch_(n_ch),
      post_filter_(post_filter_beta > 0.0f),
      post_filter_beta_(post_filter_beta),
      atten_lim_db_(atten_lim_db),
      min_db_thresh_(min_db_thresh),
      max_db_erb_thresh_(max_db_erb_thresh),
      max_db_df_thresh_(max_db_df_thresh),
      reduce_mask_(reduce_mask)
{}

RuntimeParams RuntimeParams::default_with_ch(size_t channels) {
    return RuntimeParams(
        channels,   // n_ch
        0.02f,      // post_filter_beta
        100.f,      // atten_lim_db
        -10.f,      // min_db_thresh
        30.f,       // max_db_erb_thresh
        20.f,       // max_db_df_thresh
        ReduceMask::MEAN // reduce_mask
    );
}

RuntimeParams& RuntimeParams::with_post_filter(float beta) {
    assert(beta >= 0.0f);
    if (beta > 0.0f) {
        post_filter_ = true;
    }
    post_filter_beta_ = beta;
    return *this;
}

RuntimeParams& RuntimeParams::with_atten_lim(float atten_lim_db) {
    atten_lim_db_ = atten_lim_db;
    return *this;
}

RuntimeParams& RuntimeParams::with_thresholds(float min_db_thresh,
        float max_db_erb_thresh, float max_db_df_thresh) {
    min_db_thresh_ = min_db_thresh;
    max_db_erb_thresh_ = max_db_erb_thresh;
    max_db_df_thresh_ = max_db_df_thresh;
    return *this;
}

RuntimeParams& RuntimeParams::with_mask_reduce(ReduceMask red) {
    reduce_mask_ = red;
    return *this;
}

class DfParams {
public:
    std::vector<uint8_t> enc_;
    std::vector<uint8_t> erb_dec_;
    std::vector<uint8_t> df_dec_;
    std::map<std::string, std::map<std::string, std::string>> config_;

    DfParams(const std::string& enc_path, const std::string& erb_dec_path, const std::string& df_dec_path, const std::string& config_path) {
        enc_ = read_file(enc_path);
        erb_dec_ = read_file(erb_dec_path);
        df_dec_ = read_file(df_dec_path);
        config_ = parse_config(config_path);
    }

    const std::map<std::string, std::string>& section(const std::string& section) const {
        auto it = config_.find(section);
        if (it == config_.end()) {
            throw std::runtime_error("Config section not found: " + section);
        }
        return it->second;
    }

private:
    std::vector<uint8_t> read_file(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + path);
        }
        return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    }

    std::map<std::string, std::map<std::string, std::string>> parse_config(const std::string& path) {
        std::map<std::string, std::map<std::string, std::string>> config;
        std::ifstream file(path);
        if (!file) {
            throw std::runtime_error("Failed to open config file: " + path);
        }

        std::string line;
        std::string current_section;
        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);

            // Skip empty lines or comments
            if (line.empty() || line[0] == ';' || line[0] == '#') continue;

            if (line.front() == '[' && line.back() == ']') {
                // Found section
                current_section = line.substr(1, line.size() - 2);
            } else {
                size_t delimiter = line.find('=');
                if (delimiter != std::string::npos) {
                    std::string key = line.substr(0, delimiter);
                    std::string value = line.substr(delimiter + 1);
                    key.erase(0, key.find_first_not_of(" \t"));
                    key.erase(key.find_last_not_of(" \t") + 1);
                    value.erase(0, value.find_first_not_of(" \t"));
                    value.erase(value.find_last_not_of(" \t") + 1);
                    config[current_section][key] = value;
                }
            }
        }
        return config;
    }

};


// DFNetwork class definition
class DFNetwork {
    public:
        DFNetwork(const DfParams& df_params, const RuntimeParams& rp, Ort::Env& env, Ort::SessionOptions& session_options)
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

            // Buffer allocation (using vectors for now; you could move to Ort::Value for spec_buf_/erb_buf_ if desired)
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

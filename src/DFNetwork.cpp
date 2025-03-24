#include "DFNetwork.h"
#include <string>
#include <stdexcept>
#include <iostream>
#include <utility>

TensorBuffer permute_cplx_buf(const TensorComplex& cplx_buf, size_t n_ch, size_t nb_df);
TensorBuffer OrtValueToTensorBuffer(const Ort::Value& ort_value);
void post_filter(const std::complex<float>* noisy, std::complex<float>* enh, size_t length, float beta);

float DFNetwork::process(const Eigen::MatrixXf& noisy_frame, Eigen::MatrixXf& enhanced_frame) {
    assert(noisy_frame.rows() == ch_);
    assert(noisy_frame.cols() == hop_size_);
    enhanced_frame.resize(ch_, hop_size_);

    float max_a = noisy_frame.cwiseAbs().maxCoeff();
    float rms = noisy_frame.squaredNorm() / noisy_frame.size();
    if (rms < 1e-7f) {
        skip_counter_++;
    } else {
        skip_counter_ = 0;
    }
    if (skip_counter_ > 5) {
        enhanced_frame.setZero();
        return -15.0f;   // early skip return
    }
    if (max_a > 0.9999f) {
        std::cout << "Warning: Possible clipping detected (" << max_a << ")\n";
    }

    rolling_spec_buf_y_.pop_front();
    rolling_spec_buf_x_.pop_front();

    for (size_t ch = 0; ch < ch_; ++ch) {
        // Get the noisy frame for the current channel
        auto noisy_frame_ch = noisy_frame.row(ch);
        // Get the spectral buffer for the current channel
        std::complex<float>* spec_buf_ch = spec_buf_.data.data() + ch * n_freqs_;
        // Perform analysis
        df_states_[ch].analysis(noisy_frame_ch.data(), spec_buf_ch);
    }

    rolling_spec_buf_y_.push_back(TensorComplex(spec_buf_));
    rolling_spec_buf_x_.push_back(TensorComplex(spec_buf_));

    if (atten_lim_.has_value() && atten_lim_.value() == 1.0f) {
        enhanced_frame = noisy_frame;
        return 35.0f;
    }

    auto [lsnr, gains, coefs] = process_raw();
    auto [apply_erb, gains_ignore, coefs_ignore] = apply_stages(lsnr);
    auto& spec_frame = rolling_spec_buf_y_[df_order_ - 1];
    if (gains.has_value()) {
        Eigen::Map<const Eigen::MatrixXf> gains_map(
                gains->data.data(),
                ch_,
                nb_erb_);
        if (gains_map.rows() < ch_) {
            // single channel
            df_states_[0].apply_mask(
                    Eigen::Map<Eigen::ArrayXcf>(spec_frame.data.data(), n_freqs_),
                    gains_map.row(0)
                    );
        } else {
            // multi-channel
            for (size_t ch = 0; ch < ch_; ch++) {
                df_states_[ch].apply_mask(
                        Eigen::Map<Eigen::ArrayXcf>(spec_frame.data.data() + ch * n_freqs_, n_freqs_),
                        gains_map.row(ch)
                        );
            }
        }
        skip_counter_ = 0;
    } else {
        skip_counter_++;
    }

    spec_buf_.data = spec_frame.data;
    // Apply second-stage DF filtering if coefficients present
    if (coefs.has_value()) {
        df(
                rolling_spec_buf_x_,
                *coefs,
                nb_df_,
                df_order_,
                n_freqs_,
                spec_buf_
          );
        // Get noisy spectrum snapshot for mixing
        const auto& spec_noisy_frame = rolling_spec_buf_x_[
            std::max(lookahead_, df_order_) - lookahead_ - 1
        ];
        Eigen::Map<const Eigen::ArrayXcf> spec_noisy_map(
                reinterpret_cast<const std::complex<float>*>(spec_noisy_frame.data.data()),
                ch_ * n_freqs_
                );

        // Current enhanced spectrum to work on
        Eigen::Map<Eigen::ArrayXcf> spec_enh_map(
                reinterpret_cast<std::complex<float>*>(spec_buf_.data.data()),
                ch_ * n_freqs_
                );

        // Run post filter if enabled
        if (apply_erb && post_filter_) {
            post_filter(
                    spec_noisy_map.data(),
                    spec_enh_map.data(),
                    ch_ * n_freqs_,
                    post_filter_beta_
                    );
        }

        // Apply attenuation limit if needed
        if (atten_lim_.has_value()) {
            float lim = atten_lim_.value();
            spec_enh_map = spec_enh_map * (1.0f - lim) + spec_noisy_map * lim;
        }

        // Run synthesis step per channel (IFFT)
        for (size_t ch = 0; ch < ch_; ch++) {
            auto spec_ch_data = spec_enh_map.segment(ch * n_freqs_, n_freqs_).data();
            float* enh_out_ch = enhanced_frame.row(ch).data();
            df_states_[ch].synthesis(spec_ch_data, enh_out_ch);
        }
    }
    return lsnr;
}


std::tuple<float, std::optional<TensorBuffer>, std::optional<TensorBuffer>> DFNetwork::process_raw() {
    for (size_t ch = 0; ch < ch_; ch++) {
        Eigen::Map<const Eigen::ArrayXcf> nsy_ch(spec_buf_.data.data() + ch * n_freqs_, n_freqs_);
        Eigen::Map<Eigen::ArrayXf> erb_ch(erb_buf_.data.data() + ch * nb_erb_, nb_erb_);
        Eigen::Map<Eigen::ArrayXcf> cplx_ch(cplx_buf_.data.data() + ch * nb_df_, nb_df_);

        df_states_[ch].feat_erb(nsy_ch, alpha_, erb_ch);
        df_states_[ch].feat_cplx(nsy_ch.head(nb_df_), alpha_, cplx_ch);
    }
    auto cplx_permuted_data = permute_cplx_buf(cplx_buf_, ch_, nb_df_);
    std::vector<int64_t> cplx_permuted_shape = {static_cast<int64_t>(ch_), 2, 1, static_cast<int64_t>(nb_df_)};
    // Create Ort::Value for permuted cplx_buf
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Value cplx_tensor = Ort::Value::CreateTensor<float>(
            mem_info_,
            cplx_permuted_data.data.data(),
            cplx_permuted_data.data.size(),
            cplx_permuted_data.shape.data(),
            cplx_permuted_data.shape.size()
            );

    Ort::Value erb_tensor = Ort::Value::CreateTensor<float>(
            mem_info_,                                           // CPU memory info
            erb_buf_.data.data(),                                // pointer to the erb_buf_ data
            erb_buf_.data.size(),                                // total number of elements
            erb_buf_.shape.data(),                               // pointer to erb_buf_ shape array
            erb_buf_.shape.size()                                // number of dimensions
            );

    std::array<Ort::Value, 2> encoder_inputs = {std::move(erb_tensor), std::move(cplx_tensor)};
    // Run encoder inference
    auto encoder_outputs = enc_session_.Run(
            Ort::RunOptions{nullptr},
            encoder_input_names_.data(),
            encoder_inputs.data(),
            encoder_inputs.size(),
            encoder_output_names_.data(),
            encoder_output_names_.size()
            );

    Ort::Value e0_tensor = std::move(encoder_outputs[0]);
    Ort::Value e1_tensor = std::move(encoder_outputs[1]);
    Ort::Value e2_tensor = std::move(encoder_outputs[2]);
    Ort::Value e3_tensor = std::move(encoder_outputs[3]);
    Ort::Value emb_tensor = std::move(encoder_outputs[4]);
    Ort::Value c0_tensor = std::move(encoder_outputs[5]);
    Ort::Value lsnr_tensor = std::move(encoder_outputs[6]);

    auto emb_tensor_info = emb_tensor.GetTensorTypeAndShapeInfo();
    auto emb_shape = emb_tensor_info.GetShape();
    size_t emb_num_elements = emb_tensor_info.GetElementCount();

    // Create a clone tensor from the copied data:
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value emb_tensor_clone = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(emb_tensor.GetTensorData<float>()),
            emb_num_elements,
            emb_shape.data(),
            emb_shape.size()
            );

    float lsnr = *(lsnr_tensor.GetTensorData<float>());
    auto [apply_gains, apply_gain_zeros, apply_df] = apply_stages(lsnr);
    // std::cout << "Enhancing frame with lsnr " << lsnr
    //       << " dB. Applying stage 1: " << apply_gains
    //       << " and stage 2: " << apply_df << std::endl;
    std::optional<TensorBuffer> m;
    if (apply_gains) {
        std::array<Ort::Value, 5> erb_inputs = {
            std::move(emb_tensor),
            std::move(e3_tensor),
            std::move(e2_tensor),
            std::move(e1_tensor),
            std::move(e0_tensor)
        };

        auto erb_output = erb_dec_session_.Run(
                Ort::RunOptions{nullptr},
                erb_input_names_.data(),
                erb_inputs.data(),
                erb_inputs.size(),
                erb_dec_output_names_.data(),
                1);

        // Remove unnecessary axes (this part you may handle by just taking the correct shape)
        m = OrtValueToTensorBuffer(erb_output.front());
        m->shape = {static_cast<int64_t>(ch_), static_cast<int64_t>(nb_erb_)};
    } else if (apply_gain_zeros) {
        m = TensorBuffer();
        m->data = std::vector<float>(ch_ * nb_erb_, 0.0f);
        m->shape = {static_cast<int64_t>(ch_), static_cast<int64_t>(nb_erb_)};
    } else {
        m = std::nullopt;
    }
    std::optional<TensorBuffer> coefs;
    if (apply_df) {
        std::array<Ort::Value, 2> df_inputs = {
            std::move(emb_tensor_clone),
            std::move(c0_tensor)
        };

        auto df_output = df_dec_session_.Run(Ort::RunOptions{nullptr},
                df_dec_input_names_.data(),
                df_inputs.data(),
                df_inputs.size(),
                df_dec_output_names_.data(),
                1);
        auto df_tensor_info = df_output.front().GetTensorTypeAndShapeInfo();
        coefs = OrtValueToTensorBuffer(df_output.front());
        coefs->shape = {static_cast<int64_t>(ch_), static_cast<int64_t>(nb_df_), static_cast<int64_t>(df_order_), 2};;
    }
    return {lsnr, m, coefs};
}

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

        spec_buf_.data.resize(n_freqs_ * 2, std::complex<float>(0.0f, 0.0f));
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
        TensorComplex empty;
        empty.data.assign(spec_buf_size, std::complex<float>(0.0f, 0.0f));
        empty.shape = {static_cast<int64_t>(ch_), 1, 1, static_cast<int64_t>(n_freqs_), 2};
        rolling_spec_buf_y_.clear();
        for (size_t i = 0; i < (df_order_ + lookahead_); i++) {
            rolling_spec_buf_y_.emplace_back(empty);
        }
        rolling_spec_buf_x_.clear();
        for (size_t i = 0; i < std::max(lookahead_, df_order_); i++) {
            rolling_spec_buf_x_.emplace_back(empty);
        }

        // Store runtime params
        post_filter_ = rp.post_filter_;
        post_filter_beta_ = rp.post_filter_beta_;
        min_db_thresh_ = rp.min_db_thresh_;
        max_db_erb_thresh_ = rp.max_db_erb_thresh_;
        max_db_df_thresh_ = rp.max_db_df_thresh_;
        reduce_mask_ = rp.reduce_mask_;
        skip_counter_ = 0;
        init();
    }
void DFNetwork::init() {
    size_t spec_buf_size = ch_ * 1 * 1 * n_freqs_ * 2;
    std::cout << "Initializing DFNetwork rolling buffers and state...\n";

    TensorComplex empty;
    empty.data.assign(spec_buf_size, std::complex<float>(0.0f, 0.0f));
    empty.shape = {static_cast<int64_t>(ch_), 1, 1, static_cast<int64_t>(n_freqs_), 2};
    rolling_spec_buf_y_.clear();
    for (size_t i = 0; i < (df_order_ + conv_lookahead_); ++i) {
        rolling_spec_buf_y_.emplace_back(empty);
    }

    rolling_spec_buf_x_.clear();
    for (size_t i = 0; i < std::max(df_order_, lookahead_); ++i) {
        rolling_spec_buf_x_.emplace_back(empty);
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
    spec_buf_.data.assign(spec_buf_size, std::complex<float>(0.0f, 0.0f));
    spec_buf_.shape = {static_cast<int64_t>(ch_), 1, 1, static_cast<int64_t>(n_freqs_), 2};

    erb_buf_.data.assign(ch_ * nb_erb_, 0.0f);
    erb_buf_.shape = {static_cast<int64_t>(ch_), 1, 1, static_cast<int64_t>(nb_erb_)};

    cplx_buf_.data.assign(ch_ * nb_df_ * 2, 0.0f);
    cplx_buf_.shape = {static_cast<int64_t>(ch_), 1, static_cast<int64_t>(nb_df_), 2};

    std::cout << "DFNetwork::init() completed.\n";
}

// [
//  Channel 0
//  [ [[Re0, Im0], [Re1, Im1], [Re2, Im2]] ],  // Shape [1, 3, 2]
//  Channel 1
//  [ [[Re3, Im3], [Re4, Im4], [Re5, Im5]] ]
// ]
// [
//  // Channel 0
//  [
//    [ [Re0, Re1, Re2] ],   // All real parts
//    [ [Im0, Im1, Im2] ]    // All imaginary parts
//  ],
//  // Channel 1
//  [
//    [ [Re3, Re4, Re5] ],
//    [ [Im3, Im4, Im5] ]
//  ]
//]
TensorBuffer permute_cplx_buf(const TensorComplex& cplx_buf, size_t n_ch, size_t nb_df) {
    TensorBuffer result;
    result.shape = {static_cast<int64_t>(n_ch), 2, 1, static_cast<int64_t>(nb_df)};
    result.data.resize(n_ch * 2 * 1 * nb_df);

    for (size_t ch = 0; ch < n_ch; ch++) {
        for (size_t bin = 0; bin < nb_df; bin++) {
            size_t input_idx = ch * nb_df + bin;
            const auto& val = cplx_buf.data[input_idx];

            // real part
            size_t real_idx = ch * 2 * nb_df + 0 * nb_df + bin;
            result.data[real_idx] = val.real();

            // imag part
            size_t imag_idx = ch * 2 * nb_df + 1 * nb_df + bin;
            result.data[imag_idx] = val.imag();
        }
    }
    return result;
}

std::tuple<bool, bool, bool> DFNetwork::apply_stages(float lsnr) const {
    if (lsnr < min_db_thresh_) {
        // Only noise detected, apply zero mask
        return {false, true, false};
    } else if (lsnr > max_db_erb_thresh_) {
        // Clean signal, no processing
        return {false, false, false};
    } else if (lsnr > max_db_df_thresh_) {
        // Mild noise, only apply stage 1 (erb)
        return {true, false, false};
    } else {
        // Noisy signal, apply stage 1 and 2
        return {true, false, true};
    }
}

TensorBuffer OrtValueToTensorBuffer(const Ort::Value& ort_value) {
    TensorBuffer tb;
    size_t size = ort_value.GetTensorTypeAndShapeInfo().GetElementCount();
    const float* data_ptr = ort_value.GetTensorData<float>();
    tb.data.assign(data_ptr, data_ptr + size);
    return tb;
}

void DFNetwork::df(
        const std::deque<TensorComplex>& spec,
        const TensorBuffer& coefs,
        size_t nb_df,
        size_t df_order,
        size_t n_freqs,
        TensorComplex& spec_out)
{
    // Map the output tensor as an Eigen matrix
    Eigen::Map<Eigen::MatrixXcf> o_f(spec_out.data.data(), ch_, n_freqs);
    // Zero out the first nb_df frequency bins
    o_f.leftCols(nb_df).setZero();

    assert(coefs.shape.size() == 4);
    assert(coefs.shape[0] == ch_);
    assert(coefs.shape[1] == nb_df);
    assert(coefs.shape[2] == df_order);
    assert(coefs.shape[3] == 2);

    // Iterate over the spec frames and coefficients
    for (size_t t = 0; t < df_order; ++t) {
        const auto& spec_frame = spec.at(t);
        Eigen::Map<const Eigen::MatrixXcf> spec_map(spec_frame.data.data(), ch_, n_freqs);
        for (size_t c = 0; c < ch_; ++c) {
            for (size_t f = 0; f < nb_df; ++f) {
                // Index calculation: [ch, nb_df, df_order, 0] and [ch, nb_df, df_order, 1]
                // df_order = 5
                // nb_df = 96
                size_t idx = (((c * nb_df + f) * df_order) + t) * 2;
                std::complex<float> coef_complex(
                    coefs.data[idx],
                    coefs.data[idx + 1]
                );
                // Accumulate filtered output
                o_f(c, f) += spec_map(c, f) * coef_complex;
            }
        }
    }
}

void post_filter(const std::complex<float>* noisy, std::complex<float>* enh, size_t length, float beta) {
    const float eps = 1e-12f;
    const float pi = static_cast<float>(M_PI);
    const float beta_p1 = beta + 1.0f;

    for (size_t i = 0; i < length; i += 4) {
        float g[4], g_sin[4], pf[4];

        for (int j = 0; j < 4; ++j) {
            g[j] = std::abs(enh[i + j]) / (std::abs(noisy[i + j]) + eps);
            g[j] = std::clamp(g[j], eps, 1.0f);
        }

        for (int j = 0; j < 4; ++j) {
            g_sin[j] = g[j] * std::sin(g[j] * pi * 0.5f);
            float denom = 1.0f + beta * std::pow(g[j] / (g_sin[j] + eps), 2);
            pf[j] = (beta_p1 * g[j] / denom) / g[j];
        }

        for (int j = 0; j < 4; ++j) {
            enh[i + j] *= pf[j];
        }
    }
}

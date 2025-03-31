#include "DFNetwork.h"
#include <string>
#include <stdexcept>
#include <iostream>
#include <utility>

TensorBuffer permute_cplx_buf(const TensorComplex& cplx_buf, size_t nb_df);
TensorBuffer setup_erb_input(const std::deque<TensorBuffer>& erb_rolling_buf, size_t n_ch, size_t nb_erb);
TensorBuffer setup_cplx_input(const std::deque<TensorBuffer>& cplx_rolling_buf, size_t n_ch, size_t nb_df);
void update_c0_buffer(Ort::Value& c0_tensor, std::deque<std::vector<float>>& rolling_buffer);
TensorBuffer setup_c0_buffer(const std::deque<std::vector<float>>& c0_rolling_buffer);
TensorBuffer OrtValueToTensorBuffer(const Ort::Value& ort_value);
void post_filter(const std::complex<float>* noisy, std::complex<float>* enh, size_t length, float beta);

#define NUM_DF_SAMPLES 5

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

    // Get the noisy frame for the first and only channel
    auto noisy_frame_ch = noisy_frame.row(0);
    // Perform analysis
    df_states_[0].analysis(noisy_frame_ch.data(), spec_buf_.data.data());

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
        df_states_[0].apply_mask(
                Eigen::Map<Eigen::ArrayXcf>(spec_frame.data.data(), n_freqs_),
                gains_map.row(0)
                );
        skip_counter_ = 0;
    } else {
        skip_counter_++;
    }

    spec_buf_.data = spec_frame.data;
    // Apply second-stage DF filtering if coefficients present
//    if (coefs.has_value()) {
//        df(
//                rolling_spec_buf_x_,
//                *coefs,
//                nb_df_,
//                df_order_,
//                n_freqs_,
//                spec_buf_
//          );
        // Get noisy spectrum snapshot for mixing
        const auto& spec_noisy_frame = rolling_spec_buf_x_[
            std::max(lookahead_, df_order_) - lookahead_ - 1];
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
       auto spec_ch_data = spec_enh_map.data();
       float* enh_out_ch = enhanced_frame.row(0).data();
       df_states_[0].synthesis(spec_ch_data, enh_out_ch);
//    }
    return lsnr;
}


std::tuple<float, std::optional<TensorBuffer>, std::optional<TensorBuffer>> DFNetwork::process_raw() {
    Eigen::Map<const Eigen::ArrayXcf> nsy_ch(spec_buf_.data.data(), n_freqs_);
    Eigen::Map<Eigen::ArrayXf> erb_ch(erb_buf_.data.data(), nb_erb_);
    Eigen::Map<Eigen::ArrayXcf> cplx_ch(cplx_buf_.data.data(), nb_df_);

    df_states_[0].feat_erb(nsy_ch, alpha_, erb_ch);
    df_states_[0].feat_cplx(nsy_ch.head(nb_df_), alpha_, cplx_ch);

    auto cplx_permuted_data = permute_cplx_buf(cplx_buf_, nb_df_);
    std::vector<int64_t> cplx_permuted_shape = {static_cast<int64_t>(ch_), 2, 1, static_cast<int64_t>(nb_df_)};
    rolling_erb.pop_front();
    rolling_cplx.pop_front();
    rolling_erb.push_back(erb_buf_);
    rolling_cplx.push_back(TensorBuffer(cplx_permuted_data));

    TensorBuffer erb_input = setup_erb_input(rolling_erb, ch_, nb_erb_);
    TensorBuffer cplx_input = setup_cplx_input(rolling_cplx, ch_, nb_df_);

    // Create Ort::Value for permuted cplx_buf
    Ort::Value cplx_tensor = Ort::Value::CreateTensor<float>(
            mem_info_,
            cplx_input.data.data(),
            cplx_input.data.size(),
            cplx_input.shape.data(),
            cplx_input.shape.size()
            );

    Ort::Value erb_tensor = Ort::Value::CreateTensor<float>(
            mem_info_,                                           // CPU memory info
            erb_input.data.data(),                                // pointer to the erb_buf_ data
            erb_input.data.size(),                                // total number of elements
            erb_input.shape.data(),                               // pointer to erb_buf_ shape array
            erb_input.shape.size()                                // number of dimensions
            );
    Ort::Value enc_hidden_tensor = Ort::Value::CreateTensor<float>(
            mem_info_,
            enc_hidden.data.data(),
            enc_hidden.data.size(),
            enc_hidden.shape.data(),
            enc_hidden.shape.size()
            );

    std::array<Ort::Value, 3> encoder_inputs = {std::move(erb_tensor),
        std::move(cplx_tensor), std::move(enc_hidden_tensor)};
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
    Ort::Value new_enc_hidden_tensor = std::move(encoder_outputs[7]);
    float* new_enc_hidden_data = new_enc_hidden_tensor.GetTensorMutableData<float>();
    std::copy(new_enc_hidden_data, new_enc_hidden_data + 256, enc_hidden.data.begin());

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
    std::optional<TensorBuffer> m;
    if (apply_gains) {
    Ort::Value erb_dec_hidden_tensor = Ort::Value::CreateTensor<float>(
            mem_info_,
            erb_dec_hidden.data.data(),
            erb_dec_hidden.data.size(),
            erb_dec_hidden.shape.data(),
            erb_dec_hidden.shape.size()
            );

        std::array<Ort::Value, 6> erb_inputs = {
            std::move(emb_tensor),
            std::move(e3_tensor),
            std::move(e2_tensor),
            std::move(e1_tensor),
            std::move(e0_tensor),
            std::move(erb_dec_hidden_tensor)
        };

        auto erb_output = erb_dec_session_.Run(
                Ort::RunOptions{nullptr},
                erb_input_names_.data(),
                erb_inputs.data(),
                erb_inputs.size(),
                erb_dec_output_names_.data(),
                erb_dec_output_names_.size()
                );

        m = OrtValueToTensorBuffer(erb_output[0]);
        m->shape = {1, 1, static_cast<int64_t>(ch_), static_cast<int64_t>(nb_erb_)};
        erb_dec_hidden = OrtValueToTensorBuffer(erb_output[1]);
        erb_dec_hidden.shape = {2, 1, 256};
    } else if (apply_gain_zeros) {
        m = TensorBuffer();
        m->data = std::vector<float>(ch_ * nb_erb_, 0.0f);
        m->shape = {1, 1, static_cast<int64_t>(ch_), static_cast<int64_t>(nb_erb_)};
    } else {
        m = std::nullopt;
    }
    std::optional<TensorBuffer> coefs;
    if (apply_df) {
        update_c0_buffer(c0_tensor, rolling_c0);
        TensorBuffer c0_pulse_tensor = setup_c0_buffer(rolling_c0);
        Ort::Value c0_pulse = Ort::Value::CreateTensor<float>(
                mem_info_,                                          // CPU memory info
                c0_pulse_tensor.data.data(),                        // pointer to the c0 rolling buffer data
                c0_pulse_tensor.data.size(),                        // total number of elements
                c0_pulse_tensor.shape.data(),                       // pointer to shape array
                c0_pulse_tensor.shape.size()                        // number of dimensions
                );
        Ort::Value df_dec_hidden_tensor = Ort::Value::CreateTensor<float>(
                mem_info_,
                df_dec_hidden.data.data(),
                df_dec_hidden.data.size(),
                df_dec_hidden.shape.data(),
                df_dec_hidden.shape.size()
                );

        std::array<Ort::Value, 3> df_inputs = {
            std::move(emb_tensor_clone),
            std::move(c0_pulse),
            std::move(df_dec_hidden_tensor)
        };

        auto df_output = df_dec_session_.Run(
                Ort::RunOptions{nullptr},
                df_dec_input_names_.data(),
                df_inputs.data(),
                df_inputs.size(),
                df_dec_output_names_.data(),
                df_dec_output_names_.size()
                );
        coefs = OrtValueToTensorBuffer(df_output[0]);
        coefs->shape = {static_cast<int64_t>(ch_), static_cast<int64_t>(nb_df_), static_cast<int64_t>(df_order_), 2};;
        df_dec_hidden = OrtValueToTensorBuffer(df_output[1]);
        df_dec_hidden.shape = {2, 1, 256};
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

        spec_buf_.data.resize(n_freqs_, std::complex<float>(0.0f, 0.0f));
        spec_buf_.shape = {1, 1, 1, static_cast<int64_t>(n_freqs_), 2};

        erb_buf_.data.resize(ch_ * nb_erb_, 0.0f);
        erb_buf_.shape = {static_cast<int64_t>(ch_), 1, 1, static_cast<int64_t>(nb_erb_)};

        cplx_buf_.data.resize(ch_ * nb_df_ * 2, 0.0f);
        cplx_buf_.shape = {static_cast<int64_t>(ch_), 1, static_cast<int64_t>(nb_df_), 2};
        m_zeros_.resize(nb_erb_, 0.0f);

        enc_hidden.data.resize(256, 0.0f);
        enc_hidden.shape = {1,1,256};

        erb_dec_hidden.data.resize(2 * 256, 0.0f);
        erb_dec_hidden.shape = {2, 1, 256};
        
        df_dec_hidden.data.resize(2 * 256, 0.0f);
        df_dec_hidden.shape = {2, 1, 256};

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
        size_t spec_buf_size = ch_ * 1 * 1 * n_freqs_;
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

        TensorBuffer empty_erb_buf;
        empty_erb_buf.data.assign(ch_ * nb_erb_, 0);
        empty_erb_buf.shape = {static_cast<int64_t>(ch_), static_cast<int64_t>(nb_erb_)};
        TensorBuffer empty_cplx_buf;
        empty_cplx_buf.data.assign(ch_ * nb_erb_ * 2, 0);
        empty_cplx_buf.shape = {static_cast<int64_t>(ch_), 1, static_cast<int64_t>(nb_erb_), 2};
        rolling_erb.clear();
        rolling_cplx.clear();
        // create rolling erb buffer with lookahead + 1 inputs.
        // For dfn3 this is due to (3,3) kernel
        for (int i = 0; i < lookahead_ + 1; i++) {
            rolling_erb.push_back(empty_erb_buf);
            rolling_cplx.push_back(empty_cplx_buf);
        }

        std::vector<float> empty_c0_buf;
        empty_c0_buf.assign(64 * 96, 0.0f);
        rolling_c0.clear();
        for (int i = 0; i < NUM_DF_SAMPLES; i++) {
            rolling_c0.push_back(empty_c0_buf);
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
    std::cout << "Initializing DFNetwork rolling buffers and state...\n";

    size_t spec_buf_size = ch_ * 1 * 1 * n_freqs_;
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

    TensorBuffer empty_erb_buf;
    empty_erb_buf.data.assign(ch_ * nb_erb_, 0);
    empty_erb_buf.shape = {static_cast<int64_t>(ch_), static_cast<int64_t>(nb_erb_)};
    TensorBuffer empty_cplx_buf;
    empty_cplx_buf.data.assign(ch_ * nb_erb_ * 2, 0);
    empty_cplx_buf.shape = {static_cast<int64_t>(ch_), 1, static_cast<int64_t>(nb_erb_), 2};
    rolling_erb.clear();
    rolling_cplx.clear();
    // create rolling erb buffer with lookahead + 1 inputs.
    // For dfn3 this is due to (3,3) kernel
    for (int i = 0; i < lookahead_ + 1; i++) {
        rolling_erb.push_back(empty_erb_buf);
        rolling_cplx.push_back(empty_cplx_buf);
    }
    //create rolling c0 buffer and fill with 0s first
    std::vector<float> empty_c0_buf;
    empty_c0_buf.assign(64 * 96, 0.0f);
    rolling_c0.clear();
    for (int i = 0; i < NUM_DF_SAMPLES; i++) {
        rolling_c0.push_back(empty_c0_buf);
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

    enc_hidden.data.resize(256, 0.0f);
    enc_hidden.shape = {1,1,256};

    erb_dec_hidden.data.resize(2 * 256, 0.0f);
    erb_dec_hidden.shape = {2, 1, 256};

    df_dec_hidden.data.resize(2 * 256, 0.0f);
    df_dec_hidden.shape = {2, 1, 256};
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
TensorBuffer permute_cplx_buf(const TensorComplex& cplx_buf, size_t nb_df) {
    TensorBuffer result;
    result.shape = {1, 2, 1, static_cast<int64_t>(nb_df)};
    result.data.resize(1 * 2 * 1 * nb_df);

    for (size_t bin = 0; bin < nb_df; bin++) {
        size_t input_idx = bin;
        const auto& val = cplx_buf.data[input_idx];

        // real part
        size_t real_idx = bin;
        result.data[real_idx] = val.real();

        // imag part
        size_t imag_idx = 1 * nb_df + bin;
        result.data[imag_idx] = val.imag();
    }
    return result;
}


TensorBuffer setup_erb_input(const std::deque<TensorBuffer>& erb_rolling_buf, size_t n_ch, size_t nb_erb) {
    TensorBuffer result;
    result.shape = {static_cast<int64_t>(n_ch), 1, 3, static_cast<int64_t>(nb_erb)};
    result.data.resize(n_ch * 1 * 3 * nb_erb);

    size_t offset = 0;
    for (const auto& buf : erb_rolling_buf) {
        std::copy(buf.data.begin(), buf.data.end(), result.data.begin() + offset);
        offset += buf.data.size(); // Move offset forward
    }
    return result;
}

TensorBuffer setup_cplx_input(const std::deque<TensorBuffer>& cplx_rolling_buf, size_t n_ch, size_t nb_df) {
    TensorBuffer result;
    result.shape = {static_cast<int64_t>(n_ch), 2, 3, static_cast<int64_t>(nb_df)};
    result.data.resize(n_ch * 2 * 3 * nb_df);
    size_t real_offset = 0;
    size_t imaginary_offset = 3 * nb_df;
    for (const auto& buf : cplx_rolling_buf) {
        std::copy(buf.data.begin(), buf.data.begin() + nb_df, result.data.begin() + real_offset);
        std::copy(buf.data.begin() + nb_df, buf.data.begin() + 2 * nb_df, result.data.begin() + imaginary_offset);
        real_offset += nb_df;
        imaginary_offset += nb_df;
    }
    return result;
}

void update_c0_buffer(Ort::Value& c0_tensor, std::deque<std::vector<float>>& rolling_buffer) {
    float* c0_data = c0_tensor.GetTensorMutableData<float>();
    // Convert Ort::Value to std::vector<float>
    std::vector<float> c0_vector(c0_data, c0_data + (64 * 96)); // 1 * 64 * 1 * 96
    if (rolling_buffer.size() >= NUM_DF_SAMPLES) {
        rolling_buffer.pop_front(); 
    }
    rolling_buffer.push_back(c0_vector);
}

TensorBuffer setup_c0_buffer(const std::deque<std::vector<float>>& c0_rolling_buffer) {
    TensorBuffer result;
    result.shape = {static_cast<int64_t>(1), static_cast<int64_t>(64),
        static_cast<int64_t>(NUM_DF_SAMPLES), static_cast<int64_t>(96)};
    result.data.resize(64 * NUM_DF_SAMPLES * 96);
    for (size_t t = 0; t < NUM_DF_SAMPLES; t++) {
        const std::vector<float>& timestep_buffer = c0_rolling_buffer[t];
        for (size_t i = 0; i < 64; i++) {
            size_t dest = (i * NUM_DF_SAMPLES * 96) + (t * 96);
            size_t src_idx = i * 96;
            std::copy(timestep_buffer.begin() + src_idx,
                    timestep_buffer.begin() + src_idx + 96,
                    result.data.begin() + dest);
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

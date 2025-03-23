#include <gtest/gtest.h>
#include "DFParams.h"
#include "DFNetwork.h"
#include "RuntimeParams.h"
#include <onnxruntime_cxx_api.h>


#define EXPECT_COMPLEX_EQ(expected, actual) \
    EXPECT_FLOAT_EQ((expected).real(), (actual).real()); \
    EXPECT_FLOAT_EQ((expected).imag(), (actual).imag())

const std::string MODEL_DIR = MODEL_PATH;
const std::string ENC_MODEL_PATH = MODEL_DIR + "/enc.onnx";
const std::string ERB_DEC_MODEL_PATH = MODEL_DIR + "/erb_dec.onnx";
const std::string DF_DEC_MODEL_PATH = MODEL_DIR + "/df_dec.onnx";
const std::string CONFIG_PATH = MODEL_DIR + "/config.ini";

class DFNetworkTestFixture : public ::testing::Test {
protected:
    static std::unique_ptr<DFNetwork> network_;
    static void SetUpTestSuite() {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DFNetworkTest");
        Ort::SessionOptions session_options;
        DFParams params(ENC_MODEL_PATH, ERB_DEC_MODEL_PATH, DF_DEC_MODEL_PATH, CONFIG_PATH);
        RuntimeParams rp = RuntimeParams::default_with_ch(1);
        network_ = std::make_unique<DFNetwork>(params, rp, env, session_options);
    }

    static void TearDownTestSuite() {
        network_.reset();
    }
};

std::unique_ptr<DFNetwork> DFNetworkTestFixture::network_ = nullptr;

TEST_F(DFNetworkTestFixture, SessionInitialization) {
    ASSERT_NE(network_, nullptr);
    ASSERT_GT(network_->sr_, 0);
}

TEST_F(DFNetworkTestFixture, BufferInitialization) {
    // Check rolling_spec_buf_y_
    EXPECT_EQ(network_->rolling_spec_buf_y_.size(), network_->df_order_ + network_->conv_lookahead_);
    size_t expected_size = network_->ch_ * network_->n_freqs_ * 2;  // channels * freqs * 2
    for (const auto& buf : network_->rolling_spec_buf_y_) {
        EXPECT_EQ(buf.data.size(), expected_size);
        for (std::complex<float> val : buf.data) {
            EXPECT_COMPLEX_EQ(val, std::complex<float>(0.0f, 0.0f));
        }
    }

    // Check spec_buf_
    EXPECT_EQ(network_->spec_buf_.data.size(), network_->ch_ * 1 * 1 * network_->n_freqs_ * 2);
}

TEST_F(DFNetworkTestFixture, ConstructionAndSetupSanity) {
    // Check ONNX sessions loaded
    ASSERT_GT(network_->enc_session_.GetInputCount(), 0);
    ASSERT_GT(network_->erb_dec_session_.GetInputCount(), 0);
    ASSERT_GT(network_->df_dec_session_.GetInputCount(), 0);

    // Check parameter values
    ASSERT_EQ(network_->sr_, 48000);
    ASSERT_EQ(network_->hop_size_, 480);
    ASSERT_EQ(network_->fft_size_, 960);
    ASSERT_GT(network_->nb_erb_, 0);
    ASSERT_GT(network_->nb_df_, 0);
    ASSERT_GT(network_->lookahead_, 0);

    // Check buffers are properly sized
    ASSERT_EQ(network_->spec_buf_.data.size(), network_->n_freqs_ * 2);
    ASSERT_EQ(network_->erb_buf_.data.size(), network_->ch_ * network_->nb_erb_);
    ASSERT_EQ(network_->cplx_buf_.data.size(), network_->ch_ * network_->nb_df_ * 2);
    ASSERT_EQ(network_->rolling_spec_buf_y_.size(), network_->df_order_ + network_->lookahead_);
    ASSERT_EQ(network_->rolling_spec_buf_x_.size(), std::max(network_->df_order_, network_->lookahead_));

    // Check state
    ASSERT_EQ(network_->df_states_.size(), 1);  // Should have at least one DFState
    ASSERT_GT(network_->df_states_[0].mean_norm_state_.size(), 0);
    ASSERT_GT(network_->df_states_[0].unit_norm_state_.size(), 0);

    std::cout << "DFNetwork sanity test passed.\n";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

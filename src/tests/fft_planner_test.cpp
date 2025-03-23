#include <gtest/gtest.h>
#include "RealFFTPlanner.h"
#include "FFTWRealToComplex.h"
#include "FFTWComplexToReal.h"
#include <vector>
#include <cmath>

TEST(FFTPlannerTest, ForwardFFTConstantSignal) {
    RealFFTPlanner<float> planner(
        [](size_t size) { return std::make_shared<FFTWRealToComplex<float>>(size); },
        [](size_t size) { return std::make_shared<FFTWComplexToReal<float>>(size); }
    );

    size_t size = 1024;
    auto forward_plan = planner.plan_fft_forward(size);

    std::vector<float> input(size, 1.0f);
    std::vector<std::complex<float>> freq(size / 2 + 1);
    forward_plan->process(input.data(), freq.data());

    // Check DC bin
    EXPECT_NEAR(freq[0].real(), size, 1e-3);
    EXPECT_NEAR(freq[0].imag(), 0.0, 1e-5);

    // Check other bins are near zero
    for (size_t i = 1; i < size / 2; i++) {
        EXPECT_NEAR(freq[i].real(), 0.0, 1e-4);
        EXPECT_NEAR(freq[i].imag(), 0.0, 1e-4);
    }
}

TEST(FFTPlannerTest, RoundTripAccuracy) {
    RealFFTPlanner<float> planner(
        [](size_t size) { return std::make_shared<FFTWRealToComplex<float>>(size); },
        [](size_t size) { return std::make_shared<FFTWComplexToReal<float>>(size); }
    );

    size_t size = 1024;
    auto forward_plan = planner.plan_fft_forward(size);
    auto inverse_plan = planner.plan_fft_inverse(size);

    std::vector<float> input(size);
    for (size_t i = 0; i < size; i++) {
        input[i] = sin(2 * M_PI * i / size);
    }

    std::vector<std::complex<float>> freq(size / 2 + 1);
    forward_plan->process(input.data(), freq.data());

    std::vector<float> output(size);
    inverse_plan->process(freq.data(), output.data());

    for (size_t i = 0; i < size; i++) {
        EXPECT_NEAR(output[i] / size, input[i], 1e-3);
    }
}


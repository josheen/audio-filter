#include <iostream>
#include <onnxruntime_cxx_api.h>
#include "FFTWRealToComplex.h"
#include "FFTWComplexToReal.h"
#include "RealFFTPlanner.h"

int main() {
    RealFFTPlanner<float> planner(
            [](size_t size) -> std::shared_ptr<RealToComplex<float>> {
            return std::make_shared<FFTWRealToComplex<float>>(size);
            },
            [](size_t size) -> std::shared_ptr<ComplexToReal<float>> {
            return std::make_shared<FFTWComplexToReal<float>>(size);
            }
            );
    auto forward_plan = planner.plan_fft_forward(1024);
    auto inverse_plan = planner.plan_fft_inverse(1024);

    // Forward transform:
    std::vector<float> input(1024, 1.0f);
    std::vector<std::complex<float>> freq;
    forward_plan->process(input, freq);

    // Inverse transform:
    std::vector<float> output;
    inverse_plan->process(freq, output);
    auto N = input.size();

    // Check forward FFT output:
    std::cout << "Forward FFT result (first few bins):\n";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "freq[" << i << "] = " << freq[i] << "\n";
    }
    std::cout << "Magnitude of DC bin: " << std::abs(freq[0]) << " (expected ~" << N << ")\n\n";

    // Inverse FFT verification:
    std::cout << "Inverse FFT result (first 5 samples):\n";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "output[" << i << "] = " << output[i] << "\n";
    }

    // Check if values are close to original input (within tolerance)
    float error_sum = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float expected = 1.0f;  // original input
        float diff = output[i] / N - expected; // if scaling needed
        error_sum += std::abs(diff);
    }
    std::cout << "Total error (sum of abs diffs after scaling by N): " << error_sum << "\n";
    if (error_sum / N < 1e-4)
        std::cout << "FFT round-trip check passed\n";
    else
        std::cout << "FFT round-trip check failed\n";
    return 0;
}

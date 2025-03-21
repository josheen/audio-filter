#ifndef DF_UTILS_H
#define DF_UTILS_H

#include <vector>
#include <complex>
#include <memory>


template <typename T>
class ArrayViewMut2 {
public:
    ArrayViewMut2(T* data, size_t rows, size_t cols)
        : data_(data), rows_(rows), cols_(cols) {}

    T& operator()(size_t row, size_t col) {
        return data_[row * cols_ + col];
    }

    const T& operator()(size_t row, size_t col) const {
        return data_[row * cols_ + col];
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

private:
    T* data_;
    size_t rows_;
    size_t cols_;
};

template <typename T>
class RealToComplex {
public:
    virtual ~RealToComplex() = default;
    virtual void process(const std::vector<T>& input, std::vector<std::complex<T>>& output) const = 0;
};

template <typename T>
class ComplexToReal {
public:
    virtual ~ComplexToReal() = default;
    virtual void process(const std::vector<std::vector<T>>& input, std::vector<T>& output) const = 0;
};

class DFState {
    public:
        DFState();
        virtual ~DFState() = default;
        uint32_t sampling_rate;
        uint32_t frame_size;
        uint32_t window_size;
        uint32_t freq_size;
        std::vector<float> window;
        float wnorm;
        std::vector<uint32_t> erb;
        std::vector<float> analysis_mem;
        std::vector<std::complex<float>> analysis_scratch;
        std::vector<float> synthesis_mem;
        std::vector<std::complex<float>> synthesis_scratch;
        std::vector<float> mean_norm_state;
        std::vector<float> unit_norm_state;
        std::shared_ptr<RealToComplex<float>> fft_forward;
        std::shared_ptr<ComplexToReal<float>> fft_inverse;
        using View2D = ArrayViewMut2<float>;
        float process(std::vector<float> noisy, View2D enh);
        void analysis(const std::vector<float>& input, std::vector<std::complex<float>>& output);
};

class DFonnx {
    public:
        DFonnx();
};

class DF {
    public:
        void frame_analysis(const std::vector<float>& input, std::vector<std::complex<float>>& output, DFState state);
}
#endif //  DF_UTILS_H

#ifndef FFT_LIB_H
#define FFT_LIB_H

#include <vector>
#include <complex>

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

class RealToComplexPlan {
public:
    virtual ~RealToComplexPlan() = default;
    virtual void process(const std::vector<float>& input, std::vector<std::complex<float>>& output) const = 0;
};

#endif // FFT_LIB_H

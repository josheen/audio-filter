#ifndef REAL_TO_COMPLEX_H
#define REAL_TO_COMPLEX_H
#include <vector>
#include <complex>

template <typename T>
class RealToComplex {
public:
    virtual ~RealToComplex() = default;
    virtual void process(float* input, std::complex<float>* output) const = 0;
    virtual std::unique_ptr<RealToComplex<T>> clone() const = 0;
};
#endif

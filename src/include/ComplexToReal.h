#ifndef COMPLEX_TO_REAL_H
#define COMPLEX_TO_REAL_H
#include <vector>
#include <complex>

template <typename T>
class ComplexToReal {
public:
    virtual ~ComplexToReal() = default;
    virtual void process(const std::vector<std::complex<T>>& input, std::vector<T>& output) const = 0;
    virtual std::unique_ptr<ComplexToReal<T>> clone() const = 0;
};
#endif

#include <unordered_map>
#include <memory>
#include <mutex>
#include "RealToComplex.h"
#include "ComplexToReal.h"

template <typename T>
class RealFFTPlanner {
public:
    using RealToComplexFactory = std::function<std::shared_ptr<RealToComplex<T>>(size_t)>;
    using ComplexToRealFactory = std::function<std::shared_ptr<ComplexToReal<T>>(size_t)>;

    RealFFTPlanner(RealToComplexFactory r2c_factory, ComplexToRealFactory c2r_factory)
        : r2c_factory_(std::move(r2c_factory)), c2r_factory_(std::move(c2r_factory)) {}

    std::shared_ptr<RealToComplex<T>> plan_fft_forward(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = r2c_cache_.find(size);
        if (it != r2c_cache_.end()) {
            return it->second;
        }

        auto plan = r2c_factory_(size);
        r2c_cache_[size] = plan;
        return plan;
    }

    std::shared_ptr<ComplexToReal<T>> plan_fft_inverse(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = c2r_cache_.find(size);
        if (it != c2r_cache_.end()) {
            return it->second;
        }

        auto plan = c2r_factory_(size);
        c2r_cache_[size] = plan;
        return plan;
    }

private:
    RealToComplexFactory r2c_factory_;
    ComplexToRealFactory c2r_factory_;
    std::unordered_map<size_t, std::shared_ptr<RealToComplex<T>>> r2c_cache_;
    std::unordered_map<size_t, std::shared_ptr<ComplexToReal<T>>> c2r_cache_;
    std::mutex mutex_;
};

#ifndef DF_UTILS_H
#define DF_UTILS_H

#include <vector>
#include <complex>
#include "DFState.h"

class DFonnx {
    public:
        DFonnx();
        virtual ~DFonnx() = default;
};

class DF {
    public:
        DF();
        virtual ~DF() = default;
        void frame_analysis(const std::vector<float>& input, std::vector<std::complex<float>>& output, DFState state);
};

#endif //  DF_UTILS_H

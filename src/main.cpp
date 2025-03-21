#include <iostream>
#include <onnxruntime_cxx_api.h>
#include "df_utils.h"

int main() {
    DF df;
    std::vector<float> noisy_test = {1,2,3,4,5};
    std::vector<float> buffer(3 * 4, 0.0f); // 3x4 matrix, all zero
    DF::View2D example(buffer.data(), 3, 4);
    std::cout << "process output " << df.process(noisy_test, example) << std::endl;
}

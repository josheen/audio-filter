#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        const char* model_path = "dfn2_onnx/export/enc.onnx";
        Ort::Session session(env, model_path, session_options);

        std::cout << "ONNX Runtime is working! Model loaded successfully." << std::endl;

        size_t num_inputs = session.GetInputCount();
        Ort::AllocatorWithDefaultOptions allocator;

        for (size_t i = 0; i < num_inputs; i++) {
            auto input_name = session.GetInputNameAllocated(i, allocator);
            std::cout << "Input " << i << ": " << input_name.get() << std::endl;
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

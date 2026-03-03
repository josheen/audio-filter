#include <portaudio.h>
#include <sndfile.h>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <DFNetwork.h>
#define SAMPLE_RATE 48000
#define CHANNELS 1

struct RecorderData {
    std::vector<float> inputRingBuffer;
    std::vector<float> outputBuffer;
    std::mutex bufferMutex;
    std::mutex outputMutex;
    size_t hopSize;
    DFNetwork* df;
    std::atomic<bool> running{true};
};

const std::string MODEL_DIR = MODEL_PATH;
#if NEW_MODEL
const std::string ENC_MODEL_PATH = MODEL_DIR + "/new_models/enc.onnx";
const std::string ERB_DEC_MODEL_PATH = MODEL_DIR + "/new_models/erb_dec.onnx";
const std::string DF_DEC_MODEL_PATH = MODEL_DIR + "/new_models/df_dec.onnx";
#else
const std::string ENC_MODEL_PATH = MODEL_DIR + "/enc.onnx";
const std::string ERB_DEC_MODEL_PATH = MODEL_DIR + "/erb_dec.onnx";
const std::string DF_DEC_MODEL_PATH = MODEL_DIR + "/df_dec.onnx";

#endif
const std::string CONFIG_PATH = MODEL_DIR + "/config.ini";

static int recordCallback(const void* input, void* output,
        unsigned long frameCount,
        const PaStreamCallbackTimeInfo* timeInfo,
        PaStreamCallbackFlags statusFlags,
        void* userData) {
    auto* data = static_cast<RecorderData*>(userData);
    const float* in = static_cast<const float*>(input);

    {
        std::lock_guard<std::mutex> lock(data->bufferMutex);

        // Store incoming samples
        for (unsigned long i = 0; i < frameCount; i++) {
            data->inputRingBuffer.push_back(in[i]);
        }

        while (data->inputRingBuffer.size() >= data->hopSize) {
            // Prepare input matrix
            Eigen::MatrixXf noisy(1, data->hopSize);
            for (size_t j = 0; j < data->hopSize; j++) {
                noisy(0, j) = data->inputRingBuffer[j];
            }

            // Process frame
            Eigen::MatrixXf enhanced(1, data->hopSize);
            data->df->process(noisy, enhanced);

            // Store output
            {
                std::lock_guard<std::mutex> outLock(data->outputMutex);
                for (size_t j = 0; j < data->hopSize; j++) {
                    data->outputBuffer.push_back(enhanced(0, j));
                }
            }

            // Remove processed samples
            data->inputRingBuffer.erase(data->inputRingBuffer.begin(),
                    data->inputRingBuffer.begin() + data->hopSize);
        }
    }

    return data->running ? paContinue : paComplete;
}

int main() {
    DFParams params(ENC_MODEL_PATH, ERB_DEC_MODEL_PATH, DF_DEC_MODEL_PATH, CONFIG_PATH);
    RuntimeParams rp = RuntimeParams::default_with_ch(1);
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "dfnetwork");
    Ort::SessionOptions opts;
    DFNetwork dfnet(params, rp, env, opts);
    dfnet.init();

    // Setup recorder
    RecorderData recData;
    recData.hopSize = dfnet.hop_size_;
    recData.df = &dfnet;

    Pa_Initialize();
    PaStream* stream;
    Pa_OpenDefaultStream(&stream, CHANNELS, 0, paFloat32, SAMPLE_RATE,
            recData.hopSize, recordCallback, &recData);

    std::cout << "Recording... Press enter to stop." << std::endl;
    Pa_StartStream(stream);

    // Wait for user input to stop
    std::cin.get();
    recData.running = false;

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    // Write to WAV
    SF_INFO sfinfo;
    sfinfo.samplerate = SAMPLE_RATE;
    sfinfo.channels = 1;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    SNDFILE* outfile = sf_open("enhanced_output.wav", SFM_WRITE, &sfinfo);
    if (!outfile) {
        std::cerr << "Failed to open output file.\n";
        return 1;
    }
    sf_writef_float(outfile, recData.outputBuffer.data(),
            recData.outputBuffer.size());
    sf_close(outfile);

    std::cout << "Wrote enhanced_output.wav!" << std::endl;
    return 0;
}

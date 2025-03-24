#ifndef ERB_PUB_H
#define ERB_PUB_H

#include <vector>
#include <cmath>

std::vector<size_t> erb_fb(size_t sr, size_t fft_size,size_t nb_bands, size_t min_nb_freqs);

inline float freq2erb(float freq_hz) {
    return 9.265f * std::log1p(freq_hz / (24.7f * 9.265f));
}

inline float erb2freq(float n_erb) {
    return 24.7f * 9.265f * (std::exp(n_erb / 9.265f) - 1.0f);
}
#endif

#include "ERB_pub.h"
#include <numeric>

static float freq2erb(float freq_hz);
static float erb2freq(float n_erb);

std::vector<size_t> erb_fb(size_t sr, size_t fft_size,size_t nb_bands, size_t min_nb_freqs) {
    size_t nyq_freq = sr/2;
    float freq_width = static_cast<float>(sr) / static_cast<float>(fft_size);
    float erb_low = freq2erb(0.0f);
    float erb_high = freq2erb(static_cast<float>(nyq_freq));
    
    std::vector<size_t> erb(nb_bands, 0);
    float step = (erb_high - erb_low) / static_cast<float>(nb_bands);
    
    int prev_freq = 0;
    int freq_over = 0;

    for (size_t i=1; i <= nb_bands; i++) {
        float f = erb2freq(erb_low + static_cast<float>(i) * step);
        int fb = static_cast<int>(std::round(f/freq_width));
        int nb_freqs = min_nb_freqs;
        if  (nb_freqs < min_nb_freqs) {
            freq_over = min_nb_freqs - nb_freqs;
            nb_freqs = min_nb_freqs;
        } else {
            freq_over = 0;
        }
        erb[i-1] = static_cast<size_t>(nb_freqs);
        prev_freq = fb;
    }
    erb[nb_bands - 1] += 1;
    size_t too_large = std::accumulate(erb.begin(), erb.end(), 0ul) - (fft_size / 2 + 1);
    if (too_large > 0) {
        erb[nb_bands - 1] -= too_large;
    }
    assert(std::accumulate(erb.begin(), erb.end(), 0ul) == (fft_size / 2 + 1));
    return erb;
}


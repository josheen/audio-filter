#ifndef DF_UTILS_H
#define DF_UTILS_H

#include <string>
#include <vector>

class DF {
    public:
        struct DFState {
            uint32_t sampling_rate;
            uint32_t frame_size;
            uint32_t window_size;
        };
};

#endif //  DF_UTILS_H

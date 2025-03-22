#ifndef DF_UTILS_H
#define DF_UTILS_H

#include <vector>
#include <complex>
#include <onnxruntime_cxx_api.h>
#include "DFState.h"

enum ReduceMask {
    NONE,
    MAX,
    MEAN,
};

class RuntimeParams {
    public:
        RuntimeParams(size_t n_ch, float post_filter_beta,
                float atten_lim_db, float min_db_thresh,
                float max_db_erb_thresh, float max_db_df_thresh,
                ReduceMask reduce_mask);
        size_t n_ch_;
        bool post_filter_;
        float post_filter_beta_;
        float atten_lim_db_;
        float min_db_thresh_;
        float max_db_erb_thresh_;
        float max_db_df_thresh_;
        ReduceMask reduce_mask_;

        static RuntimeParams default_with_ch(size_t channels);

        RuntimeParams& with_post_filter(float beta);
        RuntimeParams& with_atten_lim(float atten_lim_db);
        RuntimeParams& with_thresholds(float min_db_thresh,
                float max_db_erb_thresh, float max_db_df_thresh);
        RuntimeParams& with_mask_reduce(ReduceMask red);
};

#endif //  DF_UTILS_H

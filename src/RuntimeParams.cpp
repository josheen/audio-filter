#include "RuntimeParams.h"

RuntimeParams::RuntimeParams(
    size_t n_ch,
    float post_filter_beta,
    float atten_lim_db,
    float min_db_thresh,
    float max_db_erb_thresh,
    float max_db_df_thresh,
    ReduceMask reduce_mask
)
    : n_ch_(n_ch),
      post_filter_(post_filter_beta > 0.0f),
      post_filter_beta_(post_filter_beta),
      atten_lim_db_(atten_lim_db),
      min_db_thresh_(min_db_thresh),
      max_db_erb_thresh_(max_db_erb_thresh),
      max_db_df_thresh_(max_db_df_thresh),
      reduce_mask_(reduce_mask)
{}

RuntimeParams RuntimeParams::default_with_ch(size_t channels) {
    return RuntimeParams(
        channels,   // n_ch
        0.02f,      // post_filter_beta
        100.f,      // atten_lim_db
        -10.f,      // min_db_thresh
        30.f,       // max_db_erb_thresh
        20.f,       // max_db_df_thresh
        ReduceMask::MEAN // reduce_mask
    );
}

RuntimeParams& RuntimeParams::with_post_filter(float beta) {
    assert(beta >= 0.0f);
    if (beta > 0.0f) {
        post_filter_ = true;
    }
    post_filter_beta_ = beta;
    return *this;
}

RuntimeParams& RuntimeParams::with_atten_lim(float atten_lim_db) {
    atten_lim_db_ = atten_lim_db;
    return *this;
}

RuntimeParams& RuntimeParams::with_thresholds(float min_db_thresh,
        float max_db_erb_thresh, float max_db_df_thresh) {
    min_db_thresh_ = min_db_thresh;
    max_db_erb_thresh_ = max_db_erb_thresh;
    max_db_df_thresh_ = max_db_df_thresh;
    return *this;
}

RuntimeParams& RuntimeParams::with_mask_reduce(ReduceMask red) {
    reduce_mask_ = red;
    return *this;
}

// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "shortconv.h"
#include <cmath>
#include <cstring>

namespace ncnn {

ShortConv::ShortConv()
{
    one_blob_only = false;
    support_inplace = false;
}

int ShortConv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& weight_mat = bottom_blobs[0];
    const Mat& mixed_qkv = bottom_blobs[1];
    const Mat& conv_state = bottom_blobs.size() > 2 ? bottom_blobs[2] : Mat();

    int seq_len = mixed_qkv.dims == 1 ? 1 : mixed_qkv.h;
    int groups = mixed_qkv.w;
    int kernel_size = weight_mat.w;

    int conv_state_h = conv_state.empty() ? 0 : (conv_state.dims == 1 ? 1 : conv_state.h);
    int total_len = conv_state.empty() ? (kernel_size - 1 + seq_len) : (conv_state_h + seq_len);

    Mat stated_mixed_qkv;
    if (conv_state.empty())
    {
        stated_mixed_qkv.create(groups, kernel_size - 1 + seq_len, 4u, opt.workspace_allocator);
        if (stated_mixed_qkv.empty())
            return -100;
        memset(stated_mixed_qkv.row(0), 0, (kernel_size - 1) * groups * sizeof(float));
        memcpy(stated_mixed_qkv.row(kernel_size - 1), mixed_qkv.data, seq_len * groups * sizeof(float));
    }
    else
    {
        stated_mixed_qkv.create(groups, conv_state_h + seq_len, 4u, opt.workspace_allocator);
        if (stated_mixed_qkv.empty())
            return -100;
        memcpy(stated_mixed_qkv.row(0), conv_state.data, conv_state_h * groups * sizeof(float));
        memcpy(stated_mixed_qkv.row(conv_state_h), mixed_qkv.data, seq_len * groups * sizeof(float));
    }

    int state_len = kernel_size;
    if (top_blobs.size() > 1)
    {
        Mat& last_conv_state = top_blobs[1];
        last_conv_state.create(groups, state_len, 4u, opt.blob_allocator);
        if (last_conv_state.empty())
            return -100;
        memcpy(last_conv_state.data, stated_mixed_qkv.row(total_len - state_len),
               state_len * groups * sizeof(float));
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(groups, seq_len, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    int prefix_len = conv_state.empty() ? (kernel_size - 1) : conv_state_h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < groups; g++)
    {
        const float* w_ptr = weight_mat.channel(g);

        for (int i = 0; i < seq_len; i++)
        {
            float sum = 0.f;
            int base = prefix_len + i;

            for (int k = 0; k < kernel_size; k++)
            {
                int src_i = base - (kernel_size - 1) + k;
                sum += stated_mixed_qkv.row(src_i)[g] * w_ptr[k];
            }

            top_blob.row(i)[g] = sum * (1.f / (1.f + expf(-sum)));
        }
    }

    return 0;
}

} // namespace ncnn

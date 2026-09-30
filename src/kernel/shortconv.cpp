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
    if (bottom_blobs.size() < 2 || top_blobs.empty())
        return -100;

    const Mat& weight_mat = bottom_blobs[0];
    const Mat& mixed_qkv = bottom_blobs[1];
    const Mat& conv_state = bottom_blobs.size() > 2 ? bottom_blobs[2] : Mat();

    const int groups = weight_mat.c;
    const int kernel_size = weight_mat.w;
    const bool channel_major = mixed_qkv.dims != 1 && mixed_qkv.h == groups && mixed_qkv.w != groups;
    const int seq_len = mixed_qkv.dims == 1 ? 1 : channel_major ? mixed_qkv.w : mixed_qkv.h;

    if (kernel_size <= 0 || groups <= 0 || seq_len <= 0 ||
        weight_mat.dims != 3 || weight_mat.h != 1 ||
        (mixed_qkv.dims != 1 && mixed_qkv.dims != 2 && mixed_qkv.dims != 3) ||
        (mixed_qkv.dims == 1 && mixed_qkv.w != groups) ||
        (mixed_qkv.dims != 1 && mixed_qkv.w != groups &&
         (mixed_qkv.h != groups || mixed_qkv.w <= 0)) ||
        (mixed_qkv.dims == 3 && mixed_qkv.c != 1))
        return -100;

    const int prefix_len = kernel_size - 1;
    const int total_len = prefix_len + seq_len;

    bool valid_state = conv_state.empty();
    if (!valid_state)
    {
        if (conv_state.dims == 2 || (conv_state.dims == 3 && conv_state.c == 1))
        {
            if (conv_state.w == kernel_size && conv_state.h == groups)
                valid_state = true;
            else if (conv_state.w == prefix_len && conv_state.h == groups)
                valid_state = true;
            else if (conv_state.w == groups && (conv_state.h == kernel_size || conv_state.h == prefix_len))
                valid_state = true;
        }
    }
    if (!valid_state)
        return -100;

    Mat stated_mixed_qkv;
    stated_mixed_qkv.create(groups, total_len, 4u, opt.workspace_allocator);
    if (stated_mixed_qkv.empty())
        return -100;

    if (conv_state.empty())
    {
        memset(stated_mixed_qkv.data, 0, (size_t)prefix_len * groups * sizeof(float));
    }
    else if (conv_state.h == groups)
    {
        const int offset = conv_state.w - prefix_len;
        for (int t = 0; t < prefix_len; t++)
        {
            float* row = stated_mixed_qkv.row(t);
            for (int g = 0; g < groups; g++)
                row[g] = conv_state.row(g)[offset + t];
        }
    }
    else
    {
        const int offset = conv_state.h - prefix_len;
        for (int t = 0; t < prefix_len; t++)
        {
            float* row = stated_mixed_qkv.row(t);
            const float* src_row = conv_state.row(offset + t);
            memcpy(row, src_row, groups * sizeof(float));
        }
    }

    if (mixed_qkv.dims == 1)
    {
        memcpy(stated_mixed_qkv.row(prefix_len), mixed_qkv.data,
               groups * sizeof(float));
    }
    else if (channel_major)
    {
        const float* mixed_data = (const float*)mixed_qkv.data;
        for (int g = 0; g < groups; g++)
        {
            for (int t = 0; t < seq_len; t++)
                stated_mixed_qkv.row(prefix_len + t)[g] = mixed_data[g * seq_len + t];
        }
    }
    else
    {
        memcpy(stated_mixed_qkv.row(prefix_len), mixed_qkv.data,
               (size_t)seq_len * groups * sizeof(float));
    }

    if (top_blobs.size() > 1)
    {
        Mat& last_conv_state = top_blobs[1];
        last_conv_state.create(kernel_size, groups, 4u, opt.blob_allocator);
        if (last_conv_state.empty())
            return -100;
        for (int g = 0; g < groups; g++)
        {
            float* state_row = last_conv_state.row(g);
            for (int k = 0; k < kernel_size; k++)
                state_row[k] = stated_mixed_qkv.row(total_len - kernel_size + k)[g];
        }
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(channel_major ? seq_len : groups,
                    channel_major ? groups : seq_len, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < groups; g++)
    {
        const float* weight = weight_mat.channel(g);
        for (int t = 0; t < seq_len; t++)
        {
            const int base = prefix_len + t;
            float sum = 0.f;
            for (int k = 0; k < kernel_size; k++)
                sum += stated_mixed_qkv.row(base - (kernel_size - 1) + k)[g] * weight[k];
            if (channel_major)
                top_blob.row(g)[t] = sum * (1.f / (1.f + expf(-sum)));
            else
                top_blob.row(t)[g] = sum * (1.f / (1.f + expf(-sum)));
        }
    }

    return 0;
}

} // namespace ncnn

// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "shortconv_x86.h"

#if __SSE2__ || defined(__x86_64__) || defined(_M_X64)
#ifndef __SSE2__
#define __SSE2__ 1
#endif
#endif

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#if __AVX512F__
#include <immintrin.h>
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"
#include "x86_activation.h"

#include <cmath>
#include <cstring>

namespace ncnn {

ShortConv_x86::ShortConv_x86()
{
#if __SSE2__
    support_packing = false;
#endif // __SSE2__
}

int ShortConv_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
        weight_mat.dims != 3 || weight_mat.h != 1 || weight_mat.c != groups ||
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

    const bool qkv_is_1d = (mixed_qkv.dims == 1);
    const float* qkv_data = (const float*)mixed_qkv.data;

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

    if (qkv_is_1d)
    {
        memcpy(stated_mixed_qkv.row(prefix_len), qkv_data,
               groups * sizeof(float));
    }
    else if (channel_major)
    {
        for (int g = 0; g < groups; g++)
        {
            for (int t = 0; t < seq_len; t++)
                stated_mixed_qkv.row(prefix_len + t)[g] = qkv_data[g * seq_len + t];
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

        if (kernel_size == 4)
        {
            const float w0 = weight[0];
            const float w1 = weight[1];
            const float w2 = weight[2];
            const float w3 = weight[3];
            for (int t = 0; t < seq_len; t++)
            {
                const int base = prefix_len + t;
                const float value = stated_mixed_qkv.row(base - 3)[g] * w0
                          + stated_mixed_qkv.row(base - 2)[g] * w1
                          + stated_mixed_qkv.row(base - 1)[g] * w2
                          + stated_mixed_qkv.row(base)[g]     * w3;
                if (channel_major)
                    top_blob.row(g)[t] = value;
                else
                    top_blob.row(t)[g] = value;
            }
        }
        else if (kernel_size == 3)
        {
            const float w0 = weight[0];
            const float w1 = weight[1];
            const float w2 = weight[2];
            for (int t = 0; t < seq_len; t++)
            {
                const int base = prefix_len + t;
                const float value = stated_mixed_qkv.row(base - 2)[g] * w0
                          + stated_mixed_qkv.row(base - 1)[g] * w1
                          + stated_mixed_qkv.row(base)[g]     * w2;
                if (channel_major)
                    top_blob.row(g)[t] = value;
                else
                    top_blob.row(t)[g] = value;
            }
        }
        else
        {
            for (int t = 0; t < seq_len; t++)
            {
                const int base = prefix_len + t;
                float sum = 0.f;
                for (int k = 0; k < kernel_size; k++)
                    sum += stated_mixed_qkv.row(base - (kernel_size - 1) + k)[g] * weight[k];
                if (channel_major)
                    top_blob.row(g)[t] = sum;
                else
                    top_blob.row(t)[g] = sum;
            }
        }
    }

    const int total = seq_len * groups;
    float* out_ptr = (float*)top_blob.data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int idx = 0; idx < total; idx += 16)
    {
        int remain = total - idx;
        if (remain >= 16)
        {
#if __AVX512F__
            __m512 _v = _mm512_loadu_ps(out_ptr + idx);
            _mm512_storeu_ps(out_ptr + idx, swish_avx512(_v));
#else
#if __AVX__
            __m256 _v0 = _mm256_loadu_ps(out_ptr + idx);
            __m256 _v1 = _mm256_loadu_ps(out_ptr + idx + 8);
            _mm256_storeu_ps(out_ptr + idx, swish_avx(_v0));
            _mm256_storeu_ps(out_ptr + idx + 8, swish_avx(_v1));
#elif __SSE2__
            for (int k = 0; k < 16; k += 4)
            {
                __m128 _v = _mm_loadu_ps(out_ptr + idx + k);
                _mm_storeu_ps(out_ptr + idx + k, swish_sse(_v));
            }
#else
            for (int k = 0; k < 16; k++)
            {
                float val = out_ptr[idx + k];
                out_ptr[idx + k] = val * (1.f / (1.f + expf(-val)));
            }
#endif
#endif
        }
        else
        {
            for (int k = 0; k < remain; k++)
            {
                float val = out_ptr[idx + k];
                out_ptr[idx + k] = val * (1.f / (1.f + expf(-val)));
            }
        }
    }

    return 0;
}

} // namespace ncnn

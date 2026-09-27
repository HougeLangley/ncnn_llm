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
            int base = prefix_len + i;
            float sum = 0.f;

            if (kernel_size == 4)
            {
                sum = stated_mixed_qkv.row(base - 3)[g] * w_ptr[0]
                    + stated_mixed_qkv.row(base - 2)[g] * w_ptr[1]
                    + stated_mixed_qkv.row(base - 1)[g] * w_ptr[2]
                    + stated_mixed_qkv.row(base)[g]     * w_ptr[3];
            }
            else if (kernel_size == 3)
            {
                sum = stated_mixed_qkv.row(base - 2)[g] * w_ptr[0]
                    + stated_mixed_qkv.row(base - 1)[g] * w_ptr[1]
                    + stated_mixed_qkv.row(base)[g]     * w_ptr[2];
            }
            else
            {
                for (int k = 0; k < kernel_size; k++)
                {
                    int src_i = base - (kernel_size - 1) + k;
                    sum += stated_mixed_qkv.row(src_i)[g] * w_ptr[k];
                }
            }

            top_blob.row(i)[g] = sum;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < seq_len; i++)
    {
        float* out_ptr = top_blob.row(i);
        int g = 0;
#if __AVX512F__
        for (; g + 15 < groups; g += 16)
        {
            __m512 _sum = _mm512_loadu_ps(out_ptr + g);
            _mm512_storeu_ps(out_ptr + g, swish_avx512(_sum));
        }
        if (g < groups)
        {
            const unsigned int remain = groups - g;
            __mmask16 _mask = (__mmask16)((1u << remain) - 1);
            __m512 _sum = _mm512_maskz_loadu_ps(_mask, out_ptr + g);
            _mm512_mask_storeu_ps(out_ptr + g, _mask, swish_avx512(_sum));
            g = groups;
        }
#else // __AVX512F__
#if __SSE2__
#if __AVX__
        for (; g + 7 < groups; g += 8)
        {
            __m256 _sum = _mm256_loadu_ps(out_ptr + g);
            _mm256_storeu_ps(out_ptr + g, swish_avx(_sum));
        }
#endif // __AVX__
        for (; g + 3 < groups; g += 4)
        {
            __m128 _sum = _mm_loadu_ps(out_ptr + g);
            _mm_storeu_ps(out_ptr + g, swish_sse(_sum));
        }
#endif // __SSE2__
        for (; g < groups; g++)
        {
            float sum = out_ptr[g];
            out_ptr[g] = sum * (1.f / (1.f + expf(-sum)));
        }
#endif // __AVX512F__
    }

    return 0;
}

} // namespace ncnn

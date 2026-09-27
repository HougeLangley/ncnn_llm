// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gdr_x86.h"

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
#include <vector>

namespace ncnn {

GatedDeltaRule_x86::GatedDeltaRule_x86()
{
#if __SSE2__
    support_packing = false;
#endif // __SSE2__
}

static void l2norm(const float* x, float* out, int n, int dim, float eps)
{
    for (int i = 0; i < n; i++)
    {
        const float* row_in = x + i * dim;
        float* row_out = out + i * dim;

        float sum = 0.f;
        int j = 0;
#if __AVX512F__
        __m512 _sum_avx512 = _mm512_setzero_ps();
        for (; j + 15 < dim; j += 16)
        {
            __m512 _p = _mm512_loadu_ps(row_in + j);
            _sum_avx512 = _mm512_fmadd_ps(_p, _p, _sum_avx512);
        }
        sum += _mm512_reduce_add_ps(_sum_avx512);
        if (j < dim)
        {
            const unsigned int remain = dim - j;
            __mmask16 _mask = (__mmask16)((1u << remain) - 1);
            __m512 _p = _mm512_maskz_loadu_ps(_mask, row_in + j);
            __m512 _m = _mm512_mul_ps(_p, _p);
            sum += _mm512_mask_reduce_add_ps(_mask, _m);
            j = dim;
        }
#else // __AVX512F__
#if __SSE2__
#if __AVX__
        __m256 _sum_avx = _mm256_setzero_ps();
        for (; j + 7 < dim; j += 8)
        {
            __m256 _p = _mm256_loadu_ps(row_in + j);
            _sum_avx = _mm256_comp_fmadd_ps(_p, _p, _sum_avx);
        }
        __m128 lo = _mm256_castps256_ps128(_sum_avx);
        __m128 hi = _mm256_extractf128_ps(_sum_avx, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        sum += _mm_cvtss_f32(s);
#endif // __AVX__
        __m128 _sum_sse = _mm_setzero_ps();
        for (; j + 3 < dim; j += 4)
        {
            __m128 _p = _mm_loadu_ps(row_in + j);
            _sum_sse = _mm_comp_fmadd_ps(_p, _p, _sum_sse);
        }
        _sum_sse = _mm_add_ps(_sum_sse, _mm_movehl_ps(_sum_sse, _sum_sse));
        _sum_sse = _mm_add_ss(_sum_sse, _mm_shuffle_ps(_sum_sse, _sum_sse, 1));
        sum += _mm_cvtss_f32(_sum_sse);
#endif // __SSE2__
        for (; j < dim; j++)
        {
            sum += row_in[j] * row_in[j];
        }
#endif // __AVX512F__

        float inv_norm = 1.f / sqrtf(sum + eps);
        j = 0;
#if __AVX512F__
        __m512 _inv_avx512 = _mm512_set1_ps(inv_norm);
        for (; j + 15 < dim; j += 16)
        {
            __m512 _p = _mm512_loadu_ps(row_in + j);
            _mm512_storeu_ps(row_out + j, _mm512_mul_ps(_p, _inv_avx512));
        }
        if (j < dim)
        {
            const unsigned int remain = dim - j;
            __mmask16 _mask = (__mmask16)((1u << remain) - 1);
            __m512 _p = _mm512_maskz_loadu_ps(_mask, row_in + j);
            _mm512_mask_storeu_ps(row_out + j, _mask, _mm512_mul_ps(_p, _inv_avx512));
            j = dim;
        }
#else // __AVX512F__
#if __SSE2__
#if __AVX__
        __m256 _inv_avx = _mm256_set1_ps(inv_norm);
        for (; j + 7 < dim; j += 8)
        {
            __m256 _p = _mm256_loadu_ps(row_in + j);
            _mm256_storeu_ps(row_out + j, _mm256_mul_ps(_p, _inv_avx));
        }
#endif // __AVX__
        __m128 _inv_sse = _mm_set1_ps(inv_norm);
        for (; j + 3 < dim; j += 4)
        {
            __m128 _p = _mm_loadu_ps(row_in + j);
            _mm_storeu_ps(row_out + j, _mm_mul_ps(_p, _inv_sse));
        }
#endif // __SSE2__
        for (; j < dim; j++)
        {
            row_out[j] = row_in[j] * inv_norm;
        }
#endif // __AVX512F__
    }
}

static int torch_recurrent_gated_delta_rule(
    const float* query, const float* key, const float* value,
    const float* g, const float* beta,
    float* core_attn_out,
    float* last_recurrent_state,
    int batch_size, int num_heads, int seq_len,
    int k_head_dim, int v_head_dim,
    bool use_qk_l2norm_in_kernel,
    const Option& opt)
{
    Mat query_norm;
    Mat key_norm;

    const float* q_ptr = query;
    const float* k_ptr = key;

    int qk_size = batch_size * num_heads * seq_len * k_head_dim;
    if (use_qk_l2norm_in_kernel)
    {
        query_norm.create(k_head_dim, seq_len, num_heads * batch_size, 4u, opt.workspace_allocator);
        key_norm.create(k_head_dim, seq_len, num_heads * batch_size, 4u, opt.workspace_allocator);
        if (query_norm.empty() || key_norm.empty())
            return -100;

        l2norm(query, (float*)query_norm.data, batch_size * num_heads * seq_len, k_head_dim, 1e-6f);
        l2norm(key, (float*)key_norm.data, batch_size * num_heads * seq_len, k_head_dim, 1e-6f);

        q_ptr = (const float*)query_norm.data;
        k_ptr = (const float*)key_norm.data;
    }

    float scale = 1.f / sqrtf((float)k_head_dim);

    memset(core_attn_out, 0, batch_size * num_heads * seq_len * v_head_dim * sizeof(float));

    Mat kv_mem_buf(v_head_dim, num_heads, 4u, opt.workspace_allocator);
    Mat delta_buf(v_head_dim, num_heads, 4u, opt.workspace_allocator);
    if (kv_mem_buf.empty() || delta_buf.empty())
        return -100;

    float* kv_mem_buf_data = (float*)kv_mem_buf.data;
    float* delta_buf_data = (float*)delta_buf.data;

    for (int t = 0; t < seq_len; t++)
    {
        for (int b = 0; b < batch_size; b++)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int h = 0; h < num_heads; h++)
            {
                int bh_idx = (b * num_heads + h);
                const float* q_t = q_ptr + (bh_idx * seq_len + t) * k_head_dim;
                const float* k_t = k_ptr + (bh_idx * seq_len + t) * k_head_dim;
                const float* v_t = value + (bh_idx * seq_len + t) * v_head_dim;

                float g_t = g[bh_idx * seq_len + t];
                float beta_t = beta[bh_idx * seq_len + t];

                float* state = last_recurrent_state + bh_idx * k_head_dim * v_head_dim;
                float* out_t = core_attn_out + (bh_idx * seq_len + t) * v_head_dim;

                float* kv_mem = kv_mem_buf_data + h * v_head_dim;
                float* delta = delta_buf_data + h * v_head_dim;

                float g_t_exp = expf(g_t);

                // 1. Decay state
                int state_size = k_head_dim * v_head_dim;
                int idx = 0;
#if __AVX512F__
                __m512 _decay_avx512 = _mm512_set1_ps(g_t_exp);
                for (; idx + 15 < state_size; idx += 16)
                {
                    __m512 _p = _mm512_loadu_ps(state + idx);
                    _mm512_storeu_ps(state + idx, _mm512_mul_ps(_p, _decay_avx512));
                }
                if (idx < state_size)
                {
                    const unsigned int remain = state_size - idx;
                    __mmask16 _mask = (__mmask16)((1u << remain) - 1);
                    __m512 _p = _mm512_maskz_loadu_ps(_mask, state + idx);
                    _mm512_mask_storeu_ps(state + idx, _mask, _mm512_mul_ps(_p, _decay_avx512));
                    idx = state_size;
                }
#else // __AVX512F__
#if __SSE2__
#if __AVX__
                __m256 _decay_avx = _mm256_set1_ps(g_t_exp);
                for (; idx + 7 < state_size; idx += 8)
                {
                    __m256 _p = _mm256_loadu_ps(state + idx);
                    _mm256_storeu_ps(state + idx, _mm256_mul_ps(_p, _decay_avx));
                }
#endif // __AVX__
                __m128 _decay_sse = _mm_set1_ps(g_t_exp);
                for (; idx + 3 < state_size; idx += 4)
                {
                    __m128 _p = _mm_loadu_ps(state + idx);
                    _mm_storeu_ps(state + idx, _mm_mul_ps(_p, _decay_sse));
                }
#endif // __SSE2__
                for (; idx < state_size; idx++)
                {
                    state[idx] *= g_t_exp;
                }
#endif // __AVX512F__

                // 2. kv_mem[dv] = sum_dk state[dk, dv] * k_t[dk]
                memset(kv_mem, 0, v_head_dim * sizeof(float));
                for (int dk = 0; dk < k_head_dim; dk++)
                {
                    float k_val = k_t[dk];
                    const float* state_row = state + dk * v_head_dim;
                    int dv = 0;
#if __AVX512F__
                    __m512 _k_avx512 = _mm512_set1_ps(k_val);
                    for (; dv + 15 < v_head_dim; dv += 16)
                    {
                        __m512 _s = _mm512_loadu_ps(state_row + dv);
                        __m512 _m = _mm512_loadu_ps(kv_mem + dv);
                        _mm512_storeu_ps(kv_mem + dv, _mm512_fmadd_ps(_s, _k_avx512, _m));
                    }
                    if (dv < v_head_dim)
                    {
                        const unsigned int remain = v_head_dim - dv;
                        __mmask16 _mask = (__mmask16)((1u << remain) - 1);
                        __m512 _s = _mm512_maskz_loadu_ps(_mask, state_row + dv);
                        __m512 _m = _mm512_maskz_loadu_ps(_mask, kv_mem + dv);
                        _mm512_mask_storeu_ps(kv_mem + dv, _mask, _mm512_fmadd_ps(_s, _k_avx512, _m));
                        dv = v_head_dim;
                    }
#else // __AVX512F__
#if __SSE2__
#if __AVX__
                    __m256 _k_avx = _mm256_set1_ps(k_val);
                    for (; dv + 7 < v_head_dim; dv += 8)
                    {
                        __m256 _s = _mm256_loadu_ps(state_row + dv);
                        __m256 _m = _mm256_loadu_ps(kv_mem + dv);
                        _mm256_storeu_ps(kv_mem + dv, _mm256_comp_fmadd_ps(_s, _k_avx, _m));
                    }
#endif // __AVX__
                    __m128 _k_sse = _mm_set1_ps(k_val);
                    for (; dv + 3 < v_head_dim; dv += 4)
                    {
                        __m128 _s = _mm_loadu_ps(state_row + dv);
                        __m128 _m = _mm_loadu_ps(kv_mem + dv);
                        _mm_storeu_ps(kv_mem + dv, _mm_comp_fmadd_ps(_s, _k_sse, _m));
                    }
#endif // __SSE2__
                    for (; dv < v_head_dim; dv++)
                    {
                        kv_mem[dv] += state_row[dv] * k_val;
                    }
#endif // __AVX512F__
                }

                // 3. delta[dv] = (v_t[dv] - kv_mem[dv]) * beta_t
                int dv = 0;
#if __AVX512F__
                __m512 _beta_avx512 = _mm512_set1_ps(beta_t);
                for (; dv + 15 < v_head_dim; dv += 16)
                {
                    __m512 _vt = _mm512_loadu_ps(v_t + dv);
                    __m512 _km = _mm512_loadu_ps(kv_mem + dv);
                    _mm512_storeu_ps(delta + dv, _mm512_mul_ps(_mm512_sub_ps(_vt, _km), _beta_avx512));
                }
                if (dv < v_head_dim)
                {
                    const unsigned int remain = v_head_dim - dv;
                    __mmask16 _mask = (__mmask16)((1u << remain) - 1);
                    __m512 _vt = _mm512_maskz_loadu_ps(_mask, v_t + dv);
                    __m512 _km = _mm512_maskz_loadu_ps(_mask, kv_mem + dv);
                    _mm512_mask_storeu_ps(delta + dv, _mask, _mm512_mul_ps(_mm512_sub_ps(_vt, _km), _beta_avx512));
                    dv = v_head_dim;
                }
#else // __AVX512F__
#if __SSE2__
#if __AVX__
                __m256 _beta_avx = _mm256_set1_ps(beta_t);
                for (; dv + 7 < v_head_dim; dv += 8)
                {
                    __m256 _vt = _mm256_loadu_ps(v_t + dv);
                    __m256 _km = _mm256_loadu_ps(kv_mem + dv);
                    _mm256_storeu_ps(delta + dv, _mm256_mul_ps(_mm256_sub_ps(_vt, _km), _beta_avx));
                }
#endif // __AVX__
                __m128 _beta_sse = _mm_set1_ps(beta_t);
                for (; dv + 3 < v_head_dim; dv += 4)
                {
                    __m128 _vt = _mm_loadu_ps(v_t + dv);
                    __m128 _km = _mm_loadu_ps(kv_mem + dv);
                    _mm_storeu_ps(delta + dv, _mm_mul_ps(_mm_sub_ps(_vt, _km), _beta_sse));
                }
#endif // __SSE2__
                for (; dv < v_head_dim; dv++)
                {
                    delta[dv] = (v_t[dv] - kv_mem[dv]) * beta_t;
                }
#endif // __AVX512F__

                // 4. state[dk, dv] += k_t[dk] * delta[dv]
                for (int dk = 0; dk < k_head_dim; dk++)
                {
                    float k_val = k_t[dk];
                    float* state_row = state + dk * v_head_dim;
                    dv = 0;
#if __AVX512F__
                    __m512 _k_avx512 = _mm512_set1_ps(k_val);
                    for (; dv + 15 < v_head_dim; dv += 16)
                    {
                        __m512 _s = _mm512_loadu_ps(state_row + dv);
                        __m512 _d = _mm512_loadu_ps(delta + dv);
                        _mm512_storeu_ps(state_row + dv, _mm512_fmadd_ps(_k_avx512, _d, _s));
                    }
                    if (dv < v_head_dim)
                    {
                        const unsigned int remain = v_head_dim - dv;
                        __mmask16 _mask = (__mmask16)((1u << remain) - 1);
                        __m512 _s = _mm512_maskz_loadu_ps(_mask, state_row + dv);
                        __m512 _d = _mm512_maskz_loadu_ps(_mask, delta + dv);
                        _mm512_mask_storeu_ps(state_row + dv, _mask, _mm512_fmadd_ps(_k_avx512, _d, _s));
                        dv = v_head_dim;
                    }
#else // __AVX512F__
#if __SSE2__
#if __AVX__
                    __m256 _k_avx = _mm256_set1_ps(k_val);
                    for (; dv + 7 < v_head_dim; dv += 8)
                    {
                        __m256 _s = _mm256_loadu_ps(state_row + dv);
                        __m256 _d = _mm256_loadu_ps(delta + dv);
                        _mm256_storeu_ps(state_row + dv, _mm256_comp_fmadd_ps(_k_avx, _d, _s));
                    }
#endif // __AVX__
                    __m128 _k_sse = _mm_set1_ps(k_val);
                    for (; dv + 3 < v_head_dim; dv += 4)
                    {
                        __m128 _s = _mm_loadu_ps(state_row + dv);
                        __m128 _d = _mm_loadu_ps(delta + dv);
                        _mm_storeu_ps(state_row + dv, _mm_comp_fmadd_ps(_k_sse, _d, _s));
                    }
#endif // __SSE2__
                    for (; dv < v_head_dim; dv++)
                    {
                        state_row[dv] += k_val * delta[dv];
                    }
#endif // __AVX512F__
                }

                // 5. out_t[dv] = sum_dk state[dk, dv] * q_t[dk] * scale
                memset(out_t, 0, v_head_dim * sizeof(float));
                for (int dk = 0; dk < k_head_dim; dk++)
                {
                    float q_val = q_t[dk] * scale;
                    const float* state_row = state + dk * v_head_dim;
                    dv = 0;
#if __AVX512F__
                    __m512 _q_avx512 = _mm512_set1_ps(q_val);
                    for (; dv + 15 < v_head_dim; dv += 16)
                    {
                        __m512 _s = _mm512_loadu_ps(state_row + dv);
                        __m512 _o = _mm512_loadu_ps(out_t + dv);
                        _mm512_storeu_ps(out_t + dv, _mm512_fmadd_ps(_s, _q_avx512, _o));
                    }
                    if (dv < v_head_dim)
                    {
                        const unsigned int remain = v_head_dim - dv;
                        __mmask16 _mask = (__mmask16)((1u << remain) - 1);
                        __m512 _s = _mm512_maskz_loadu_ps(_mask, state_row + dv);
                        __m512 _o = _mm512_maskz_loadu_ps(_mask, out_t + dv);
                        _mm512_mask_storeu_ps(out_t + dv, _mask, _mm512_fmadd_ps(_s, _q_avx512, _o));
                        dv = v_head_dim;
                    }
#else // __AVX512F__
#if __SSE2__
#if __AVX__
                    __m256 _q_avx = _mm256_set1_ps(q_val);
                    for (; dv + 7 < v_head_dim; dv += 8)
                    {
                        __m256 _s = _mm256_loadu_ps(state_row + dv);
                        __m256 _o = _mm256_loadu_ps(out_t + dv);
                        _mm256_storeu_ps(out_t + dv, _mm256_comp_fmadd_ps(_s, _q_avx, _o));
                    }
#endif // __AVX__
                    __m128 _q_sse = _mm_set1_ps(q_val);
                    for (; dv + 3 < v_head_dim; dv += 4)
                    {
                        __m128 _s = _mm_loadu_ps(state_row + dv);
                        __m128 _o = _mm_loadu_ps(out_t + dv);
                        _mm_storeu_ps(out_t + dv, _mm_comp_fmadd_ps(_s, _q_sse, _o));
                    }
#endif // __SSE2__
                    for (; dv < v_head_dim; dv++)
                    {
                        out_t[dv] += state_row[dv] * q_val;
                    }
#endif // __AVX512F__
                }
            }
        }
    }

    return 0;
}

int GatedDeltaRule_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& A_log = bottom_blobs[0];
    const Mat& dt_bias = bottom_blobs[1];
    const Mat& b = bottom_blobs[2];
    const Mat& a = bottom_blobs[3];
    const Mat& query = bottom_blobs[4];
    const Mat& key = bottom_blobs[5];
    const Mat& value = bottom_blobs[6];
    const Mat& initial_state = bottom_blobs.size() > 7 ? bottom_blobs[7] : Mat();

    int num_heads = query.h;
    int seq_len = query.c;
    int k_head_dim = query.w;
    int v_head_dim = value.w;

    bool use_qk_l2norm_in_kernel = true;

    Mat& top_blob = top_blobs[0];
    top_blob.create(k_head_dim, num_heads, seq_len, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    Mat state_out_workspace;
    Mat& state_out = (top_blobs.size() > 1) ? top_blobs[1] : state_out_workspace;
    if (top_blobs.size() > 1)
    {
        state_out.create(v_head_dim, k_head_dim, num_heads, 4u, opt.blob_allocator);
    }
    else
    {
        state_out.create(v_head_dim, k_head_dim, num_heads, 4u, opt.workspace_allocator);
    }
    if (state_out.empty())
        return -100;

    const float* query_data = (const float*)query.data;
    const float* key_data = (const float*)key.data;
    const float* value_data = (const float*)value.data;

    Mat query_t(k_head_dim, seq_len, num_heads, 4u, opt.workspace_allocator);
    Mat key_t(k_head_dim, seq_len, num_heads, 4u, opt.workspace_allocator);
    Mat value_t(v_head_dim, seq_len, num_heads, 4u, opt.workspace_allocator);
    if (query_t.empty() || key_t.empty() || value_t.empty())
        return -100;

    float* query_t_data = (float*)query_t.data;
    float* key_t_data = (float*)key_t.data;
    float* value_t_data = (float*)value_t.data;

    if (seq_len == 1)
    {
        memcpy(query_t_data, query_data, num_heads * k_head_dim * sizeof(float));
        memcpy(key_t_data, key_data, num_heads * k_head_dim * sizeof(float));
        memcpy(value_t_data, value_data, num_heads * v_head_dim * sizeof(float));
    }
    else
    {
        for (int t = 0; t < seq_len; t++)
        {
            for (int h = 0; h < num_heads; h++)
            {
                int src_k = (t * num_heads + h) * k_head_dim;
                int dst_k = (h * seq_len + t) * k_head_dim;
                memcpy(query_t_data + dst_k, query_data + src_k, k_head_dim * sizeof(float));
                memcpy(key_t_data + dst_k, key_data + src_k, k_head_dim * sizeof(float));

                int src_v = (t * num_heads + h) * v_head_dim;
                int dst_v = (h * seq_len + t) * v_head_dim;
                memcpy(value_t_data + dst_v, value_data + src_v, v_head_dim * sizeof(float));
            }
        }
    }

    const float* b_data = (const float*)b.data;
    const float* a_data = (const float*)a.data;
    const float* A_log_data = (const float*)A_log.data;
    const float* dt_bias_data = (const float*)dt_bias.data;

    Mat beta(seq_len, num_heads, 4u, opt.workspace_allocator);
    Mat g(seq_len, num_heads, 4u, opt.workspace_allocator);
    if (beta.empty() || g.empty())
        return -100;

    float* beta_data = (float*)beta.data;
    float* g_data = (float*)g.data;

    for (int h = 0; h < num_heads; h++)
    {
        for (int t = 0; t < seq_len; t++)
        {
            float b_val = b_data[t * num_heads + h];
            beta_data[h * seq_len + t] = 1.f / (1.f + expf(-b_val));
        }
    }

    for (int h = 0; h < num_heads; h++)
    {
        float A_log_val = A_log_data[h];
        float dt_bias_val = dt_bias_data[h];
        float exp_A = expf(A_log_val);

        for (int t = 0; t < seq_len; t++)
        {
            float a_val = a_data[t * num_heads + h];
            float x = a_val + dt_bias_val;
            float sp_val;
            if (x > 20.f)
                sp_val = x;
            else if (x < -20.f)
                sp_val = expf(x);
            else
                sp_val = logf(1.f + expf(x));
            g_data[h * seq_len + t] = -exp_A * sp_val;
        }
    }

    int batch_size = 1;

    Mat core_attn_out(v_head_dim, seq_len, num_heads, 4u, opt.workspace_allocator);
    if (core_attn_out.empty())
        return -100;

    float* state_data = (float*)state_out.data;
    if (!initial_state.empty())
    {
        memcpy(state_data, initial_state.data,
               num_heads * k_head_dim * v_head_dim * sizeof(float));
    }
    else
    {
        memset(state_data, 0,
               num_heads * k_head_dim * v_head_dim * sizeof(float));
    }

    int ret = torch_recurrent_gated_delta_rule(
        query_t_data, key_t_data, value_t_data,
        g_data, beta_data,
        (float*)core_attn_out.data,
        state_data,
        batch_size, num_heads, seq_len,
        k_head_dim, v_head_dim,
        use_qk_l2norm_in_kernel,
        opt);
    if (ret != 0)
        return ret;

    float* top_data = (float*)top_blob.data;
    const float* core_attn_out_data = (const float*)core_attn_out.data;
    if (seq_len == 1)
    {
        memcpy(top_data, core_attn_out_data, num_heads * v_head_dim * sizeof(float));
    }
    else
    {
        for (int h = 0; h < num_heads; h++)
        {
            for (int t = 0; t < seq_len; t++)
            {
                int src_idx = (h * seq_len + t) * v_head_dim;
                int dst_idx = (t * num_heads + h) * v_head_dim;
                memcpy(top_data + dst_idx, core_attn_out_data + src_idx, v_head_dim * sizeof(float));
            }
        }
    }

    return 0;
}

} // namespace ncnn

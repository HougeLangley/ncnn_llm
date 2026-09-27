// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gdr.h"
#include <cmath>
#include <cstring>

namespace ncnn {

static void l2norm(const float* x, float* out, int n, int dim, float eps)
{
    for (int i = 0; i < n; i++)
    {
        const float* row_in = x + i * dim;
        float* row_out = out + i * dim;

        float sum = 0.f;
        for (int j = 0; j < dim; j++)
        {
            sum += row_in[j] * row_in[j];
        }
        float inv_norm = 1.f / sqrtf(sum + eps);

        for (int j = 0; j < dim; j++)
        {
            row_out[j] = row_in[j] * inv_norm;
        }
    }
}

static float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

static float softplusf(float x)
{
    if (x > 20.f)
        return x;
    if (x < -20.f)
        return expf(x);
    return logf(1.f + expf(x));
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

    Mat kv_mem(v_head_dim, 4u, opt.workspace_allocator);
    Mat delta(v_head_dim, 4u, opt.workspace_allocator);
    if (kv_mem.empty() || delta.empty())
        return -100;

    float* kv_mem_ptr = (float*)kv_mem.data;
    float* delta_ptr = (float*)delta.data;

    for (int t = 0; t < seq_len; t++)
    {
        for (int b = 0; b < batch_size; b++)
        {
            for (int h = 0; h < num_heads; h++)
            {
                const float* q_t = q_ptr + ((b * num_heads + h) * seq_len + t) * k_head_dim;
                const float* k_t = k_ptr + ((b * num_heads + h) * seq_len + t) * k_head_dim;
                const float* v_t = value + ((b * num_heads + h) * seq_len + t) * v_head_dim;

                float g_t = g[(b * num_heads + h) * seq_len + t];
                float beta_t = beta[(b * num_heads + h) * seq_len + t];

                float* state = last_recurrent_state + (b * num_heads + h) * k_head_dim * v_head_dim;
                float* out_t = core_attn_out + ((b * num_heads + h) * seq_len + t) * v_head_dim;

                float g_t_exp = expf(g_t);

                for (int i = 0; i < k_head_dim * v_head_dim; i++)
                {
                    state[i] *= g_t_exp;
                }

                memset(kv_mem_ptr, 0, v_head_dim * sizeof(float));
                for (int dv = 0; dv < v_head_dim; dv++)
                {
                    for (int dk = 0; dk < k_head_dim; dk++)
                    {
                        kv_mem_ptr[dv] += state[dk * v_head_dim + dv] * k_t[dk];
                    }
                }

                for (int dv = 0; dv < v_head_dim; dv++)
                {
                    delta_ptr[dv] = (v_t[dv] - kv_mem_ptr[dv]) * beta_t;
                }

                for (int dk = 0; dk < k_head_dim; dk++)
                {
                    for (int dv = 0; dv < v_head_dim; dv++)
                    {
                        state[dk * v_head_dim + dv] += k_t[dk] * delta_ptr[dv];
                    }
                }

                for (int dv = 0; dv < v_head_dim; dv++)
                {
                    float sum = 0.f;
                    for (int dk = 0; dk < k_head_dim; dk++)
                    {
                        sum += state[dk * v_head_dim + dv] * q_t[dk] * scale;
                    }
                    out_t[dv] = sum;
                }
            }
        }
    }

    return 0;
}

GatedDeltaRule::GatedDeltaRule()
{
    one_blob_only = false;
    support_inplace = false;

    num_k_heads = 128;
    num_v_heads = 128;
}

int GatedDeltaRule::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

    for (int t = 0; t < seq_len; t++)
    {
        for (int h = 0; h < num_heads; h++)
        {
            for (int d = 0; d < k_head_dim; d++)
            {
                int src_idx = (t * num_heads + h) * k_head_dim + d;
                int dst_idx = (h * seq_len + t) * k_head_dim + d;
                query_t_data[dst_idx] = query_data[src_idx];
                key_t_data[dst_idx] = key_data[src_idx];
            }
            for (int d = 0; d < v_head_dim; d++)
            {
                int src_idx = (t * num_heads + h) * v_head_dim + d;
                int dst_idx = (h * seq_len + t) * v_head_dim + d;
                value_t_data[dst_idx] = value_data[src_idx];
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
            beta_data[h * seq_len + t] = sigmoidf(b_val);
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
            float sp_val = softplusf(a_val + dt_bias_val);
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

    const float* core_attn_out_data = (const float*)core_attn_out.data;
    float* top_data = (float*)top_blob.data;
    for (int h = 0; h < num_heads; h++)
    {
        for (int t = 0; t < seq_len; t++)
        {
            for (int d = 0; d < v_head_dim; d++)
            {
                int src_idx = (h * seq_len + t) * v_head_dim + d;
                int dst_idx = (t * num_heads + h) * v_head_dim + d;
                top_data[dst_idx] = core_attn_out_data[src_idx];
            }
        }
    }

    return 0;
}

} // namespace ncnn

#if __SSE2__ || defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include "kernel/x86/gdr_x86.h"
#include "kernel/x86/shortconv_x86.h"
#endif

namespace ncnn {

static bool s_force_naive = false;

static Layer* GatedDeltaRule_creator(void*)
{
#if __SSE2__ || defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if (!s_force_naive)
        return new GatedDeltaRule_x86;
#endif
    return new GatedDeltaRule;
}

static void GatedDeltaRule_destroyer(Layer* layer, void*)
{
    delete layer;
}

static Layer* ShortConv_creator(void*)
{
#if __SSE2__ || defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if (!s_force_naive)
        return new ShortConv_x86;
#endif
    return new ShortConv;
}

static void ShortConv_destroyer(Layer* layer, void*)
{
    delete layer;
}

} // namespace ncnn

void register_gdr_layers(ncnn::Net& net, bool force_naive)
{
    ncnn::s_force_naive = force_naive;
    net.register_custom_layer("GatedDeltaRule", ncnn::GatedDeltaRule_creator, ncnn::GatedDeltaRule_destroyer);
    net.register_custom_layer("ShortConv", ncnn::ShortConv_creator, ncnn::ShortConv_destroyer);
}

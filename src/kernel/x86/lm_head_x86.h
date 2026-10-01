#pragma once

#include <cstddef>

namespace ncnn_llm {

// High-performance GEMV (M=1) for LM Head
// logits[i] = dot(x, weight[i]) for i in [0, N)

// Case 1: x is FP32, weight is FP32
void gemv_fp32_x86(const float* x, const float* weight, float* logits,
                   int N, int K, int num_threads);

// Case 2: x is BF16, weight is BF16
void gemv_bf16_x86(const unsigned short* x, const unsigned short* weight, float* logits,
                   int N, int K, int num_threads);

// Case 3: x is FP32, weight is BF16
void gemv_bf16_fp32_x86(const float* x, const unsigned short* weight, float* logits,
                        int N, int K, int num_threads);

} // namespace ncnn_llm

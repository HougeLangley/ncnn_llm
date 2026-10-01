#include <vector>
#include "kernel/x86/lm_head_x86.h"

#include <immintrin.h>
#include <omp.h>
#include <cstring>
#include <algorithm>

namespace ncnn_llm {

static inline float horizontal_sum_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

static inline __m256 bfloat2float_avx2(__m128i u16) {
    __m256i u32 = _mm256_cvtepu16_epi32(u16);
    return _mm256_castsi256_ps(_mm256_slli_epi32(u32, 16));
}

static inline float bfloat2float_scalar(unsigned short u16) {
    unsigned int u32 = (unsigned int)u16 << 16;
    float f;
    std::memcpy(&f, &u32, sizeof(float));
    return f;
}

void gemv_fp32_x86(const float* x, const float* weight, float* logits,
                   int N, int K, int num_threads) {
    if (N <= 0 || K <= 0 || !x || !weight || !logits) return;

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < N; i += 8) {
        if (i + 7 < N) {
            const float* w0 = weight + (size_t)(i + 0) * K;
            const float* w1 = weight + (size_t)(i + 1) * K;
            const float* w2 = weight + (size_t)(i + 2) * K;
            const float* w3 = weight + (size_t)(i + 3) * K;
            const float* w4 = weight + (size_t)(i + 4) * K;
            const float* w5 = weight + (size_t)(i + 5) * K;
            const float* w6 = weight + (size_t)(i + 6) * K;
            const float* w7 = weight + (size_t)(i + 7) * K;

            int k = 0;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();
            __m256 acc5 = _mm256_setzero_ps();
            __m256 acc6 = _mm256_setzero_ps();
            __m256 acc7 = _mm256_setzero_ps();

            for (; k + 7 < K; k += 8) {
                __m256 vx = _mm256_loadu_ps(x + k);
                acc0 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w0 + k), acc0);
                acc1 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w1 + k), acc1);
                acc2 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w2 + k), acc2);
                acc3 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w3 + k), acc3);
                acc4 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w4 + k), acc4);
                acc5 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w5 + k), acc5);
                acc6 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w6 + k), acc6);
                acc7 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w7 + k), acc7);
            }

            float s0 = horizontal_sum_avx2(acc0);
            float s1 = horizontal_sum_avx2(acc1);
            float s2 = horizontal_sum_avx2(acc2);
            float s3 = horizontal_sum_avx2(acc3);
            float s4 = horizontal_sum_avx2(acc4);
            float s5 = horizontal_sum_avx2(acc5);
            float s6 = horizontal_sum_avx2(acc6);
            float s7 = horizontal_sum_avx2(acc7);

            for (; k < K; k++) {
                const float vx = x[k];
                s0 += vx * w0[k];
                s1 += vx * w1[k];
                s2 += vx * w2[k];
                s3 += vx * w3[k];
                s4 += vx * w4[k];
                s5 += vx * w5[k];
                s6 += vx * w6[k];
                s7 += vx * w7[k];
            }

            logits[i + 0] = s0;
            logits[i + 1] = s1;
            logits[i + 2] = s2;
            logits[i + 3] = s3;
            logits[i + 4] = s4;
            logits[i + 5] = s5;
            logits[i + 6] = s6;
            logits[i + 7] = s7;
        } else {
            for (int rem = i; rem < N; rem++) {
                const float* wr = weight + (size_t)rem * K;
                int k = 0;
                __m256 acc0 = _mm256_setzero_ps();
                for (; k + 7 < K; k += 8) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k), _mm256_loadu_ps(wr + k), acc0);
                }
                float s = horizontal_sum_avx2(acc0);
                for (; k < K; k++) {
                    s += x[k] * wr[k];
                }
                logits[rem] = s;
            }
        }
    }
}

void gemv_bf16_fp32_x86(const float* x, const unsigned short* weight, float* logits,
                        int N, int K, int num_threads) {
    if (N <= 0 || K <= 0 || !x || !weight || !logits) return;

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < N; i += 8) {
        if (i + 7 < N) {
            const unsigned short* w0 = weight + (size_t)(i + 0) * K;
            const unsigned short* w1 = weight + (size_t)(i + 1) * K;
            const unsigned short* w2 = weight + (size_t)(i + 2) * K;
            const unsigned short* w3 = weight + (size_t)(i + 3) * K;
            const unsigned short* w4 = weight + (size_t)(i + 4) * K;
            const unsigned short* w5 = weight + (size_t)(i + 5) * K;
            const unsigned short* w6 = weight + (size_t)(i + 6) * K;
            const unsigned short* w7 = weight + (size_t)(i + 7) * K;

            int k = 0;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();
            __m256 acc5 = _mm256_setzero_ps();
            __m256 acc6 = _mm256_setzero_ps();
            __m256 acc7 = _mm256_setzero_ps();

            for (; k + 7 < K; k += 8) {
                __m256 vx = _mm256_loadu_ps(x + k);
                acc0 = _mm256_fmadd_ps(vx, bfloat2float_avx2(_mm_loadu_si128((const __m128i*)(w0 + k))), acc0);
                acc1 = _mm256_fmadd_ps(vx, bfloat2float_avx2(_mm_loadu_si128((const __m128i*)(w1 + k))), acc1);
                acc2 = _mm256_fmadd_ps(vx, bfloat2float_avx2(_mm_loadu_si128((const __m128i*)(w2 + k))), acc2);
                acc3 = _mm256_fmadd_ps(vx, bfloat2float_avx2(_mm_loadu_si128((const __m128i*)(w3 + k))), acc3);
                acc4 = _mm256_fmadd_ps(vx, bfloat2float_avx2(_mm_loadu_si128((const __m128i*)(w4 + k))), acc4);
                acc5 = _mm256_fmadd_ps(vx, bfloat2float_avx2(_mm_loadu_si128((const __m128i*)(w5 + k))), acc5);
                acc6 = _mm256_fmadd_ps(vx, bfloat2float_avx2(_mm_loadu_si128((const __m128i*)(w6 + k))), acc6);
                acc7 = _mm256_fmadd_ps(vx, bfloat2float_avx2(_mm_loadu_si128((const __m128i*)(w7 + k))), acc7);
            }

            float s0 = horizontal_sum_avx2(acc0);
            float s1 = horizontal_sum_avx2(acc1);
            float s2 = horizontal_sum_avx2(acc2);
            float s3 = horizontal_sum_avx2(acc3);
            float s4 = horizontal_sum_avx2(acc4);
            float s5 = horizontal_sum_avx2(acc5);
            float s6 = horizontal_sum_avx2(acc6);
            float s7 = horizontal_sum_avx2(acc7);

            for (; k < K; k++) {
                const float vx = x[k];
                s0 += vx * bfloat2float_scalar(w0[k]);
                s1 += vx * bfloat2float_scalar(w1[k]);
                s2 += vx * bfloat2float_scalar(w2[k]);
                s3 += vx * bfloat2float_scalar(w3[k]);
                s4 += vx * bfloat2float_scalar(w4[k]);
                s5 += vx * bfloat2float_scalar(w5[k]);
                s6 += vx * bfloat2float_scalar(w6[k]);
                s7 += vx * bfloat2float_scalar(w7[k]);
            }

            logits[i + 0] = s0;
            logits[i + 1] = s1;
            logits[i + 2] = s2;
            logits[i + 3] = s3;
            logits[i + 4] = s4;
            logits[i + 5] = s5;
            logits[i + 6] = s6;
            logits[i + 7] = s7;
        } else {
            for (int rem = i; rem < N; rem++) {
                const unsigned short* wr = weight + (size_t)rem * K;
                int k = 0;
                __m256 acc0 = _mm256_setzero_ps();
                for (; k + 7 < K; k += 8) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k), bfloat2float_avx2(_mm_loadu_si128((const __m128i*)(wr + k))), acc0);
                }
                float s = horizontal_sum_avx2(acc0);
                for (; k < K; k++) {
                    s += x[k] * bfloat2float_scalar(wr[k]);
                }
                logits[rem] = s;
            }
        }
    }
}

void gemv_bf16_x86(const unsigned short* x, const unsigned short* weight, float* logits,
                   int N, int K, int num_threads) {
    if (N <= 0 || K <= 0 || !x || !weight || !logits) return;

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }

    std::vector<float> x_fp32(K);
    int k = 0;
    for (; k + 7 < K; k += 8) {
        __m256 vx = bfloat2float_avx2(_mm_loadu_si128((const __m128i*)(x + k)));
        _mm256_storeu_ps(x_fp32.data() + k, vx);
    }
    for (; k < K; k++) {
        x_fp32[k] = bfloat2float_scalar(x[k]);
    }

    gemv_bf16_fp32_x86(x_fp32.data(), weight, logits, N, K, num_threads);
}

} // namespace ncnn_llm

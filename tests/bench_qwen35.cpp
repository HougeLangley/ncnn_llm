#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <benchmark.h>
#include "ncnn_llm_gpt.h"
#include "kernel/gdr.h"

struct BenchStats {
    double prefill_ms = 0;
    int prefill_tokens = 0;
    double prefill_tps = 0;
    double decode_ms = 0;
    int decode_tokens = 0;
    double decode_tps = 0;
    double ms_per_token = 0;
    std::string text;
};

BenchStats run_inference(const std::string& model_path, int threads, bool force_naive, int max_new_tokens = 32) {
    BenchStats stats;

    // Keep the requested custom-layer dispatch for the model's decoder net.
    ncnn_llm_gpt model(model_path, false, threads, 0, false, force_naive);

    std::string prompt = "<|im_start|>user\n请用一句话简短介绍人工智能。<|im_end|>\n<|im_start|>assistant\n";

    auto t0 = ncnn::get_current_time();
    auto ctx = model.prefill(prompt);
    auto t1 = ncnn::get_current_time();

    stats.prefill_ms = t1 - t0;
    stats.prefill_tokens = 16; // approximate prompt token count
    stats.prefill_tps = stats.prefill_tokens * 1000.0 / stats.prefill_ms;

    GenerateConfig cfg;
    cfg.max_new_tokens = max_new_tokens;
    cfg.temperature = 0.0f;
    cfg.do_sample = 0;

    int tokens = 0;
    std::string out;
    auto t2 = ncnn::get_current_time();
    model.generate(ctx, cfg, [&](const std::string& token) {
        tokens++;
        out += token;
    });
    auto t3 = ncnn::get_current_time();

    stats.decode_ms = t3 - t2;
    stats.decode_tokens = tokens;
    stats.decode_tps = tokens > 0 ? (tokens * 1000.0 / stats.decode_ms) : 0;
    stats.ms_per_token = tokens > 0 ? (stats.decode_ms / tokens) : 0;
    stats.text = out;

    return stats;
}

int main() {
    std::cout << "================================================================================\n";
    std::cout << "                 Qwen3.5-0.8B End-to-End Speed Benchmark                        \n";
    std::cout << "================================================================================\n\n";

    std::vector<int> thread_configs = {4, 8};

    for (int threads : thread_configs) {
        std::cout << "--------------------------------------------------------------------------------\n";
        std::cout << ">>> Running with " << threads << " CPU Threads <<<\n";
        std::cout << "--------------------------------------------------------------------------------\n";

        // 1. FP32 Model: Naive vs x86
        std::cout << "\n[Model: qwen3.5_0.8b (FP32)]\n";
        std::cout << "  1) Naive (Pure C++ / Scalar GDR & ShortConv)... " << std::flush;
        auto fp32_naive = run_inference("./assets/qwen3.5_0.8b", threads, true, 32);
        std::cout << "Done.\n";
        std::cout << "     Prefill: " << std::fixed << std::setprecision(2) << fp32_naive.prefill_ms << " ms\n";
        std::cout << "     Decode:  " << std::fixed << std::setprecision(2) << fp32_naive.decode_tps << " tokens/s (" 
                  << fp32_naive.ms_per_token << " ms/token)\n";

        std::cout << "  2) x86 SIMD Optimized (SSE2/AVX/AVX512 + ncnn intrinsics)... " << std::flush;
        auto fp32_x86 = run_inference("./assets/qwen3.5_0.8b", threads, false, 32);
        std::cout << "Done.\n";
        std::cout << "     Prefill: " << std::fixed << std::setprecision(2) << fp32_x86.prefill_ms << " ms ("
                  << (fp32_naive.prefill_ms / fp32_x86.prefill_ms) << "x speedup)\n";
        std::cout << "     Decode:  " << std::fixed << std::setprecision(2) << fp32_x86.decode_tps << " tokens/s (" 
                  << fp32_x86.ms_per_token << " ms/token, "
                  << (fp32_x86.decode_tps / fp32_naive.decode_tps) << "x speedup)\n";
        std::cout << "     Sample Output: " << fp32_x86.text.substr(0, 60) << "...\n";

        // 2. INT8 Model: Naive vs x86
        std::cout << "\n[Model: qwen3.5_0.8b_int8 (Block-Quantized INT8)]\n";
        std::cout << "  1) Naive (Pure C++ / Scalar GDR & ShortConv)... " << std::flush;
        auto int8_naive = run_inference("./assets/qwen3.5_0.8b_int8", threads, true, 32);
        std::cout << "Done.\n";
        std::cout << "     Prefill: " << std::fixed << std::setprecision(2) << int8_naive.prefill_ms << " ms\n";
        std::cout << "     Decode:  " << std::fixed << std::setprecision(2) << int8_naive.decode_tps << " tokens/s (" 
                  << int8_naive.ms_per_token << " ms/token)\n";

        std::cout << "  2) x86 SIMD Optimized (SSE2/AVX/AVX512 + ncnn intrinsics)... " << std::flush;
        auto int8_x86 = run_inference("./assets/qwen3.5_0.8b_int8", threads, false, 32);
        std::cout << "Done.\n";
        std::cout << "     Prefill: " << std::fixed << std::setprecision(2) << int8_x86.prefill_ms << " ms ("
                  << (int8_naive.prefill_ms / int8_x86.prefill_ms) << "x speedup)\n";
        std::cout << "     Decode:  " << std::fixed << std::setprecision(2) << int8_x86.decode_tps << " tokens/s (" 
                  << int8_x86.ms_per_token << " ms/token, "
                  << (int8_x86.decode_tps / int8_naive.decode_tps) << "x speedup)\n";
        std::cout << "     Sample Output: " << int8_x86.text.substr(0, 60) << "...\n";
    }

    return 0;
}

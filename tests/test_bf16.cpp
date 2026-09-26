#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

#include <net.h>
#include <benchmark.h>

#include "ncnn_llm_gpt.h"
#include "ncnn_embedding.h"

struct RunResult {
    bool success;
    std::string error;
    double prefill_ms;
    double decode_ms;
    int token_count;
    double decode_tps;
    std::string response;
};

RunResult test_inference(const std::string& model_path, const std::string& prompt, bool use_bf16) {
    RunResult r;
    r.success = false;
    r.error = "";
    r.prefill_ms = 0;
    r.decode_ms = 0;
    r.token_count = 0;
    r.decode_tps = 0;
    r.response = "";
    try {
        ncnn_llm_gpt model(model_path, false, 8, 0, use_bf16);

        auto t0 = ncnn::get_current_time();
        auto ctx = model.prefill(prompt);
        auto t1 = ncnn::get_current_time();

        GenerateConfig cfg;
        cfg.max_new_tokens = 32;
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

        r.success = true;
        r.prefill_ms = t1 - t0;
        r.decode_ms = t3 - t2;
        r.token_count = tokens;
        r.decode_tps = tokens > 0 ? (tokens * 1000.0 / r.decode_ms) : 0;
        r.response = out;
    } catch (const std::exception& e) {
        r.success = false;
        r.error = e.what();
    } catch (...) {
        r.success = false;
        r.error = "unknown exception";
    }
    return r;
}

int main() {
    std::vector<std::pair<std::string, std::string>> models = {
        {"qwen3_0.6b", "./assets/qwen3_0.6b"},
        {"minicpm4_0.5b", "./assets/minicpm4_0.5b"},
        {"qwen3.5_0.8b", "./assets/qwen3.5_0.8b"},
        {"youtu_llm", "./assets/youtu_llm"}
    };

    std::string prompt = "你好，请用一句话介绍你自己。";

    std::cout << "========================================================================\n";
    std::cout << "               Testing Text LLMs: FP32 vs BF16 Storage                  \n";
    std::cout << "========================================================================\n\n";

    for (const auto& pair : models) {
        const auto& name = pair.first;
        const auto& path = pair.second;
        std::cout << ">>> Testing Model: " << name << " <<<\n";

        std::cout << "  [FP32] Running...\n";
        auto res_fp32 = test_inference(path, prompt, false);
        if (!res_fp32.success) {
            std::cout << "  [FP32] FAILED: " << res_fp32.error << "\n";
        } else {
            std::cout << "  [FP32] Success (" << res_fp32.prefill_ms << " ms prefill, "
                      << res_fp32.decode_tps << " tokens/s)\n"
                      << "  [FP32 Output]: " << res_fp32.response << "\n";
        }

        std::cout << "  [BF16] Running...\n";
        auto res_bf16 = test_inference(path, prompt, true);
        if (!res_bf16.success) {
            std::cout << "  [BF16] FAILED: " << res_bf16.error << "\n";
        } else {
            std::cout << "  [BF16] Success (" << res_bf16.prefill_ms << " ms prefill, "
                      << res_bf16.decode_tps << " tokens/s)\n"
                      << "  [BF16 Output]: " << res_bf16.response << "\n";
        }

        if (res_fp32.success && res_bf16.success) {
            double prefill_spd = (res_fp32.prefill_ms - res_bf16.prefill_ms) / res_fp32.prefill_ms * 100.0;
            double decode_spd = (res_bf16.decode_tps - res_fp32.decode_tps) / res_fp32.decode_tps * 100.0;
            std::cout << "  => Speedup: Prefill " << std::fixed << std::setprecision(1) << prefill_spd
                      << "%, Decode " << decode_spd << "%\n";
        }
        std::cout << "------------------------------------------------------------------------\n\n";
    }

    return 0;
}

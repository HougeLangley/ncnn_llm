#include "perf_profiler.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace ncnn_llm {

static int g_perf_level = -1;

void set_perf_level(int level) {
    g_perf_level = level;
#if NCNN_BENCHMARK
    ncnn::set_layer_benchmark_active(level >= 2);
#endif
}

int get_perf_level() {
    if (g_perf_level == -1) {
        const char* env = std::getenv("NCNN_LLM_PERF");
        if (env) {
            int val = std::atoi(env);
            if (val > 0) {
                g_perf_level = val;
            } else if (std::strcmp(env, "true") == 0 || std::strcmp(env, "on") == 0) {
                g_perf_level = 1;
            } else {
                g_perf_level = 0;
            }
        } else {
            g_perf_level = 0;
        }
#if NCNN_BENCHMARK
        if (g_perf_level >= 2) {
            ncnn::set_layer_benchmark_active(true);
        }
#endif
    }
    return g_perf_level;
}

bool is_perf_enabled() {
    return get_perf_level() > 0;
}

std::string LlmPerfReport::analyze_bottlenecks() const {
    std::ostringstream ss;
    ss << "=== BOTTLENECK DIAGNOSIS & OBSERVATIONS ===\n";

    double total_wall = prefill.total_ms + decode.total_ms;
    if (total_wall <= 0) {
        ss << "- No timing data recorded.\n";
        return ss.str();
    }

    if (prefill.total_ms > 0) {
        double pf_pct = prefill.total_ms / total_wall * 100.0;
        ss << "- Prefill Phase: " << std::fixed << std::setprecision(1) << prefill.total_ms << " ms ("
           << pf_pct << "% of total, " << prefill.tokens_per_second() << " tok/s)\n";

        // Find primary prefill bottleneck
        const PhaseStat* top_pf = nullptr;
        for (const auto* p : prefill.all_phases()) {
            if (!top_pf || p->total_ms > top_pf->total_ms) {
                top_pf = p;
            }
        }
        if (top_pf && top_pf->total_ms > 0) {
            double p_pct = top_pf->total_ms / prefill.total_ms * 100.0;
            ss << "  * Largest prefill consumer: [" << top_pf->name << "] at "
               << top_pf->total_ms << " ms (" << p_pct << "% of prefill)\n";
        }
    }

    if (decode.total_ms > 0) {
        double dc_pct = decode.total_ms / total_wall * 100.0;
        ss << "- Decode Phase: " << std::fixed << std::setprecision(1) << decode.total_ms << " ms ("
           << dc_pct << "% of total, " << decode.tokens_per_second() << " tok/s, "
           << decode.ms_per_token() << " ms/tok)\n";

        // Find primary decode bottleneck
        const PhaseStat* top_dc = nullptr;
        for (const auto* p : decode.all_phases()) {
            if (!top_dc || p->total_ms > top_dc->total_ms) {
                top_dc = p;
            }
        }
        if (top_dc && top_dc->total_ms > 0) {
            double d_pct = top_dc->total_ms / decode.total_ms * 100.0;
            ss << "  * Largest decode consumer: [" << top_dc->name << "] at "
               << top_dc->total_ms << " ms (" << d_pct << "% of decode, "
               << (top_dc->count > 0 ? top_dc->total_ms / top_dc->count : 0.0) << " ms/tok)\n";
        }
    }

#if NCNN_BENCHMARK
    if (has_layer_stats && !layer_type_stats.empty()) {
        double total_layer_ms = 0;
        for (const auto& ts : layer_type_stats) total_layer_ms += ts.total_ms;
        if (total_layer_ms > 0) {
            const auto& top_type = layer_type_stats[0];
            double top_pct = top_type.total_ms / total_layer_ms * 100.0;
            ss << "- Operator Bottleneck inside Decoder Net:\n";
            ss << "  * Top layer type: [" << top_type.type << "] accounts for "
               << std::fixed << std::setprecision(1) << top_type.total_ms << " ms ("
               << top_pct << "% of all layer compute, " << top_type.count << " calls)\n";
            if (top_type.type == "Gemm") {
                ss << "  * Suggestion: Gemm operations dominate. Consider block-quantized INT8 weights, AVX2/AVX512 GEMM optimization, or thread pool adjustments.\n";
            } else if (top_type.type == "SDPA") {
                ss << "  * Suggestion: Attention compute is high. Verify FlashAttention / KV cache reuse kernel path.\n";
            }
        }
    }
#endif

    return ss.str();
}

std::string LlmPerfReport::to_string(bool include_layer_stats) const {
    std::ostringstream ss;

    ss << "================================================================================\n";
    ss << "                       NCNN-LLM INFERENCE PERF PROFILE                          \n";
    ss << "================================================================================\n";
    if (!meta.model_path.empty()) {
        ss << "Model: " << meta.model_path;
        if (!meta.model_type.empty()) ss << " (" << meta.model_type << ")";
        ss << " | Threads: " << meta.num_threads;
        ss << " | Vulkan: " << (meta.use_vulkan ? "ON" : "OFF");
        ss << " | BF16: " << (meta.use_bf16 ? "ON" : "OFF") << "\n";
    }

    double total_ms = prefill.total_ms + decode.total_ms;
    int total_tokens = prefill.prompt_tokens + decode.decode_tokens;
    ss << "Overall: " << total_tokens << " tokens in " << std::fixed << std::setprecision(2)
       << total_ms << " ms | Prefill: " << prefill.prompt_tokens << " tok ("
       << prefill.tokens_per_second() << " tok/s) | Decode: " << decode.decode_tokens
       << " tok (" << decode.tokens_per_second() << " tok/s, " << decode.ms_per_token() << " ms/tok)\n";

    if (prefill.prompt_tokens > 0 && prefill.total_ms > 0) {
        ss << "--------------------------------------------------------------------------------\n";
        ss << "PREFILL BREAKDOWN (" << prefill.prompt_tokens << " prompt tokens, "
           << std::fixed << std::setprecision(2) << prefill.total_ms << " ms total)\n";
        ss << "Phase                               Count   Total (ms)   Avg/Tok (ms)   Time %\n";
        ss << "---------------------------------   -----   ----------   ------------   ------\n";

        for (const auto* p : prefill.all_phases()) {
            if (p->count == 0 && p->total_ms == 0.0) continue;
            double pct = prefill.total_ms > 0 ? (p->total_ms / prefill.total_ms * 100.0) : 0.0;
            double avg_tok = prefill.prompt_tokens > 0 ? (p->total_ms / prefill.prompt_tokens) : 0.0;
            ss << std::left << std::setw(34) << p->name
               << std::right << std::setw(6) << p->count << "   "
               << std::fixed << std::setprecision(2) << std::setw(10) << p->total_ms << "   "
               << std::fixed << std::setprecision(3) << std::setw(12) << avg_tok << "   "
               << std::fixed << std::setprecision(1) << std::setw(5) << pct << "%\n";
        }
        ss << "---------------------------------   -----   ----------   ------------   ------\n";
        ss << std::left << std::setw(34) << "Prefill Total"
           << std::right << std::setw(6) << "-" << "   "
           << std::fixed << std::setprecision(2) << std::setw(10) << prefill.total_ms << "   "
           << std::fixed << std::setprecision(3) << std::setw(12)
           << (prefill.prompt_tokens > 0 ? prefill.total_ms / prefill.prompt_tokens : 0.0) << "   "
           << "100.0%\n";
    }

    if (decode.decode_tokens > 0 && decode.total_ms > 0) {
        ss << "--------------------------------------------------------------------------------\n";
        ss << "DECODE BREAKDOWN (" << decode.decode_tokens << " generated tokens, "
           << std::fixed << std::setprecision(2) << decode.total_ms << " ms total, "
           << decode.ms_per_token() << " ms/tok)\n";
        ss << "Phase                               Count   Total (ms)   Avg/Tok (ms)   Time %\n";
        ss << "---------------------------------   -----   ----------   ------------   ------\n";

        for (const auto* p : decode.all_phases()) {
            if (p->count == 0 && p->total_ms == 0.0) continue;
            double pct = decode.total_ms > 0 ? (p->total_ms / decode.total_ms * 100.0) : 0.0;
            double avg_tok = decode.decode_tokens > 0 ? (p->total_ms / decode.decode_tokens) : 0.0;
            ss << std::left << std::setw(34) << p->name
               << std::right << std::setw(6) << p->count << "   "
               << std::fixed << std::setprecision(2) << std::setw(10) << p->total_ms << "   "
               << std::fixed << std::setprecision(3) << std::setw(12) << avg_tok << "   "
               << std::fixed << std::setprecision(1) << std::setw(5) << pct << "%\n";
        }
        ss << "---------------------------------   -----   ----------   ------------   ------\n";
        ss << std::left << std::setw(34) << "Decode Total"
           << std::right << std::setw(6) << decode.decode_tokens << "   "
           << std::fixed << std::setprecision(2) << std::setw(10) << decode.total_ms << "   "
           << std::fixed << std::setprecision(3) << std::setw(12) << decode.ms_per_token() << "   "
           << "100.0%\n";
    }

#if NCNN_BENCHMARK
    if (include_layer_stats && has_layer_stats && !layer_type_stats.empty()) {
        double total_layer_time = 0.0;
        for (const auto& ts : layer_type_stats) total_layer_time += ts.total_ms;

        ss << "--------------------------------------------------------------------------------\n";
        ss << "DECODER LAYER BREAKDOWN BY OP TYPE (Inside Transformer Net)\n";
        ss << "Layer Type              Calls      Total (ms)      Avg (ms)     Time %\n";
        ss << "---------------------   -------   ------------   ------------   ------\n";
        for (const auto& ts : layer_type_stats) {
            double pct = total_layer_time > 0 ? (ts.total_ms / total_layer_time * 100.0) : 0.0;
            double avg = ts.count > 0 ? (ts.total_ms / ts.count) : 0.0;
            ss << std::left << std::setw(22) << ts.type
               << std::right << std::setw(8) << ts.count << "   "
               << std::fixed << std::setprecision(2) << std::setw(12) << ts.total_ms << "   "
               << std::fixed << std::setprecision(3) << std::setw(12) << avg << "   "
               << std::fixed << std::setprecision(1) << std::setw(5) << pct << "%\n";
        }
        ss << "---------------------   -------   ------------   ------------   ------\n";
        ss << std::left << std::setw(22) << "Layer Time Total"
           << std::right << std::setw(8) << "-" << "   "
           << std::fixed << std::setprecision(2) << std::setw(12) << total_layer_time << "   "
           << std::right << std::setw(12) << "-" << "   "
           << "100.0%\n";

        if (!layer_stats.empty()) {
            int show_cnt = std::min((int)layer_stats.size(), 12);
            ss << "--------------------------------------------------------------------------------\n";
            ss << "TOP " << show_cnt << " SLOWEST INDIVIDUAL LAYERS:\n";
            ss << " #   Layer Name                     Type              Total (ms)   Avg (ms)   Calls\n";
            ss << "---  -----------------------------  ----------------  ----------   --------   -----\n";
            for (int i = 0; i < show_cnt; i++) {
                const auto& ls = layer_stats[i];
                double avg = ls.count > 0 ? (ls.total_ms / ls.count) : 0.0;
                ss << std::right << std::setw(2) << (i + 1) << "   "
                   << std::left << std::setw(30) << ls.name.substr(0, 29) << " "
                   << std::left << std::setw(17) << ls.type.substr(0, 16) << " "
                   << std::right << std::fixed << std::setprecision(2) << std::setw(10) << ls.total_ms << "   "
                   << std::fixed << std::setprecision(3) << std::setw(8) << avg << "   "
                   << std::setw(5) << ls.count << "\n";
            }
        }
    }
#endif

    ss << "--------------------------------------------------------------------------------\n";
    ss << analyze_bottlenecks();
    ss << "================================================================================\n";

    return ss.str();
}

void LlmPerfReport::print(bool include_layer_stats) const {
    std::cerr << to_string(include_layer_stats);
}

nlohmann::json LlmPerfReport::to_json(bool include_layer_stats) const {
    nlohmann::json j;
    j["model"] = {
        {"path", meta.model_path},
        {"type", meta.model_type},
        {"num_threads", meta.num_threads},
        {"use_vulkan", meta.use_vulkan},
        {"use_bf16", meta.use_bf16}
    };

    j["prefill"] = {
        {"prompt_tokens", prefill.prompt_tokens},
        {"total_ms", prefill.total_ms},
        {"tokens_per_second", prefill.tokens_per_second()}
    };
    for (const auto* p : prefill.all_phases()) {
        j["prefill"]["phases"][p->name] = {
            {"count", p->count},
            {"total_ms", p->total_ms},
            {"avg_ms", p->avg_ms()},
            {"min_ms", p->min_ms < 1e8 ? p->min_ms : 0.0},
            {"max_ms", p->max_ms}
        };
    }

    j["decode"] = {
        {"decode_tokens", decode.decode_tokens},
        {"total_ms", decode.total_ms},
        {"tokens_per_second", decode.tokens_per_second()},
        {"ms_per_token", decode.ms_per_token()}
    };
    for (const auto* p : decode.all_phases()) {
        j["decode"]["phases"][p->name] = {
            {"count", p->count},
            {"total_ms", p->total_ms},
            {"avg_ms", p->avg_ms()},
            {"min_ms", p->min_ms < 1e8 ? p->min_ms : 0.0},
            {"max_ms", p->max_ms}
        };
    }

#if NCNN_BENCHMARK
    if (include_layer_stats && has_layer_stats) {
        nlohmann::json types_arr = nlohmann::json::array();
        for (const auto& ts : layer_type_stats) {
            types_arr.push_back({
                {"type", ts.type},
                {"count", ts.count},
                {"total_ms", ts.total_ms},
                {"min_ms", ts.min_ms},
                {"max_ms", ts.max_ms}
            });
        }
        j["layer_types"] = types_arr;

        nlohmann::json layers_arr = nlohmann::json::array();
        for (const auto& ls : layer_stats) {
            layers_arr.push_back({
                {"name", ls.name},
                {"type", ls.type},
                {"count", ls.count},
                {"total_ms", ls.total_ms},
                {"min_ms", ls.min_ms},
                {"max_ms", ls.max_ms}
            });
        }
        j["layers"] = layers_arr;
    }
#endif

    return j;
}

} // namespace ncnn_llm

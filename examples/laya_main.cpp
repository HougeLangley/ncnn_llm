// Copyright (c) 2026 ncnn_llm authors. All rights reserved.
// Use of this source code is governed by a BSD-style license.

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include <benchmark.h>
#include "ncnn_llm_laya.h"
#include "utf8_args.h"


static double run_bench_masked_sdpa(int L, int num_heads, int head_dim, int window, int num_threads, int iters) {
    ncnn::Option opt;
    opt.num_threads = num_threads;
    int half_win = window / 2;

    ncnn::Mat q(head_dim, L, num_heads);
    ncnn::Mat k(head_dim, L, num_heads);
    ncnn::Mat v(head_dim, L, num_heads);
    q.fill(0.01f);
    k.fill(0.01f);
    v.fill(0.01f);

    ncnn::Mat mask(L, L, 1);
    for (int i = 0; i < L; ++i) {
        float* row = mask.row(i);
        for (int j = 0; j < L; ++j) {
            row[j] = (std::abs(i - j) <= half_win) ? 0.0f : -10000.0f;
        }
    }

    ncnn::Layer* sdpa = ncnn::create_layer("SDPA");
    ncnn::ParamDict pd;
    pd.set(5, 1);
    pd.set(6, 1.0f / std::sqrt((float)head_dim));
    sdpa->load_param(pd);

    std::vector<ncnn::Mat> bottoms = {q, k, v, mask};
    std::vector<ncnn::Mat> tops(1);

    for (int i = 0; i < 5; ++i) {
        sdpa->forward(bottoms, tops, opt);
    }

    auto t0 = ncnn::get_current_time();
    for (int it = 0; it < iters; ++it) {
        sdpa->forward(bottoms, tops, opt);
    }
    auto t1 = ncnn::get_current_time();

    delete sdpa;
    return (t1 - t0) / iters;
}

static double run_bench_true_sliding(int L, int num_heads, int head_dim, int window, int num_threads, int iters) {
    int half_win = window / 2;
    float scale = 1.0f / std::sqrt((float)head_dim);

    ncnn::Mat q(head_dim, L, num_heads);
    ncnn::Mat k(head_dim, L, num_heads);
    ncnn::Mat v(head_dim, L, num_heads);
    q.fill(0.01f);
    k.fill(0.01f);
    v.fill(0.01f);

    ncnn::Mat top(head_dim, L, num_heads);

    auto run_kernel = [&]() {
        #pragma omp parallel for num_threads(num_threads)
        for (int h = 0; h < num_heads; ++h) {
            const ncnn::Mat q_head = q.channel(h);
            const ncnn::Mat k_head = k.channel(h);
            const ncnn::Mat v_head = v.channel(h);
            ncnn::Mat out_head = top.channel(h);

            std::vector<float> scores(window + 1);

            for (int i = 0; i < L; ++i) {
                const float* q_ptr = q_head.row(i);
                int j_start = std::max(0, i - half_win);
                int j_end = std::min(L, i + half_win + 1);
                int win_len = j_end - j_start;

                float max_val = -1e9f;
                for (int w_idx = 0; w_idx < win_len; ++w_idx) {
                    int j = j_start + w_idx;
                    const float* k_ptr = k_head.row(j);
                    float sum = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        sum += q_ptr[d] * k_ptr[d];
                    }
                    float sc = sum * scale;
                    scores[w_idx] = sc;
                    if (sc > max_val) max_val = sc;
                }

                float sum_exp = 0.0f;
                for (int w_idx = 0; w_idx < win_len; ++w_idx) {
                    scores[w_idx] = std::exp(scores[w_idx] - max_val);
                    sum_exp += scores[w_idx];
                }
                float inv_sum = 1.0f / sum_exp;

                float* out_ptr = out_head.row(i);
                for (int d = 0; d < head_dim; ++d) {
                    float val = 0.0f;
                    for (int w_idx = 0; w_idx < win_len; ++w_idx) {
                        int j = j_start + w_idx;
                        val += (scores[w_idx] * inv_sum) * v_head.row(j)[d];
                    }
                    out_ptr[d] = val;
                }
            }
        }
    };

    for (int i = 0; i < 5; ++i) {
        run_kernel();
    }

    auto t0 = ncnn::get_current_time();
    for (int it = 0; it < iters; ++it) {
        run_kernel();
    }
    auto t1 = ncnn::get_current_time();

    return (t1 - t0) / iters;
}

static void run_sliding_benchmark(int num_threads) {
    std::cout << "\n========================================================================================\n";
    std::cout << "  Sliding Window Attention Benchmark: Masked SDPA (O(N^2)) vs True Sliding Window (O(N*W))\n";
    std::cout << "========================================================================================\n";
    std::cout << "Config: 16 heads, 64 head_dim, Window = 128 (half_win = 64), " << num_threads << " threads\n\n";

    std::vector<int> lens = {64, 128, 192, 256, 384, 512};

    std::cout << std::left
              << std::setw(8)  << "SeqLen"
              << std::setw(22) << "Masked SDPA (ms)"
              << std::setw(26) << "True Sliding (ms)"
              << std::setw(18) << "Speedup / Ratio"
              << "Observation\n";
    std::cout << "----------------------------------------------------------------------------------------\n";

    for (int L : lens) {
        int iters = (L <= 128) ? 50 : 25;
        double t_masked = run_bench_masked_sdpa(L, 16, 64, 128, num_threads, iters);
        double t_true = run_bench_true_sliding(L, 16, 64, 128, num_threads, iters);

        double speedup = t_masked / t_true;

        std::cout << std::left
                  << std::setw(8)  << L
                  << std::setw(22) << std::fixed << std::setprecision(4) << t_masked
                  << std::setw(26) << std::fixed << std::setprecision(4) << t_true
                  << std::setw(18) << std::fixed << std::setprecision(2) << speedup;

        if (speedup >= 1.25) {
            std::cout << "True Sliding is " << std::setprecision(1) << speedup << "x faster\n";
        } else if (speedup <= 0.8) {
            std::cout << "Masked SDPA is " << std::setprecision(1) << (1.0 / speedup) << "x faster (dense loop)\n";
        } else {
            std::cout << "Comparable performance\n";
        }
    }
    std::cout << "========================================================================================\n\n";
}

int main(int argc, char** argv) {
    enable_utf8_console();
    std::vector<std::string> args = get_utf8_args(argc, argv);

    std::string model_path = "assets/laya";
    bool use_vulkan = false;
    int num_threads = 4;
    bool use_bf16 = true;
    std::string state_override = "";
    std::string json_file = "";

    for (size_t i = 1; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--model" || arg == "-m") {
            if (i + 1 < args.size()) model_path = args[++i];
        } else if (arg == "--threads" || arg == "-t") {
            if (i + 1 < args.size()) num_threads = std::stoi(args[++i]);
        } else if (arg == "--bench" || arg == "--benchmark") {
            run_sliding_benchmark(num_threads);
            return 0;
        } else if (arg == "--vulkan") {
            use_vulkan = true;
        } else if (arg == "--fp32") {
            use_bf16 = false;
        } else if (arg == "--bf16") {
            use_bf16 = true;
        } else if (arg == "--state" || arg == "-s") {
            if (i + 1 < args.size()) state_override = args[++i];
        } else if (arg == "--json" || arg == "-j") {
            if (i + 1 < args.size()) json_file = args[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: laya_main [options]\n"
                      << "Options:\n"
                      << "  -m, --model <path>    Model path (default: assets/laya or assets/laya_int8)\n"
                      << "  -t, --threads <n>     Number of threads (default: 4)\n"
                      << "  --vulkan              Enable Vulkan GPU acceleration\n"
                      << "  --bf16                Enable BF16 storage (default)\n"
                      << "  --fp32                Use FP32 storage\n"
                      << "  -s, --state <text>    Input document / state string\n"
                      << "  -j, --json <file>     Input Jev-compatible JSON request file\n"
                      << "  -h, --help            Show this help message\n";
            return 0;
        } else if (i == 1 && arg[0] != '-') {
            model_path = arg;
        }
    }

    std::cout << "========================================================================\n";
    std::cout << "               Laya Discriminator (System 1 Decision Engine)            \n";
    std::cout << "========================================================================\n";
    std::cout << "Model path: " << model_path << "\n";
    std::cout << "Threads:    " << num_threads << "\n";
    std::cout << "Precision:  " << (use_bf16 ? "BF16" : "FP32") << "\n";
    std::cout << "Vulkan:     " << (use_vulkan ? "Enabled" : "Disabled") << "\n\n";

    ncnn_llm_laya laya(model_path, use_vulkan, num_threads, 0, use_bf16);
    if (!laya.ok()) {
        std::cerr << "Failed to initialize Laya model from " << model_path << std::endl;
        return 1;
    }

    if (!json_file.empty()) {
        std::ifstream ifs(json_file);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open JSON request file: " << json_file << std::endl;
            return 1;
        }
        nlohmann::json req;
        ifs >> req;

        std::string state = req.value("state", "");
        nlohmann::json questions = req["questions"];

        auto t0 = ncnn::get_current_time();
        nlohmann::json res = laya.system_one_json(state, questions);
        auto t1 = ncnn::get_current_time();

        std::cout << "\n--- Result (JSON) ---\n";
        std::cout << res.dump(2) << std::endl;
        std::cout << "\nInference time: " << (t1 - t0) << " ms\n";
        return 0;
    }

    std::string state = state_override.empty() ?
        "Customer: My package has not arrived yet and tracking says delayed for 5 days. Agent: I apologize for the delay. Let me check the shipment." :
        state_override;

    std::map<std::string, LayaQuestion> questions;

    // 1. Choice question (Sentiment analysis)
    LayaQuestion q1;
    q1.type = "choice";
    q1.instructions = "What is the customer sentiment?";
    q1.choice_criteria = {
        {"positive", "satisfied or happy"},
        {"neutral", "neutral inquiry"},
        {"negative", "frustrated or unhappy"}
    };
    questions["q_sentiment"] = q1;

    // 2. Score question (Urgency level 0..3)
    LayaQuestion q2;
    q2.type = "score";
    q2.instructions = "How urgent is this issue?";
    q2.score_criteria = {
        "low",
        "medium",
        "high",
        "critical"
    };
    questions["q_urgency"] = q2;

    // 3. Noul question (Refund hypothesis)
    LayaQuestion q3;
    q3.type = "noul";
    q3.instructions = "Does the customer request a refund?";
    q3.noul_false = "no refund mentioned";
    q3.noul_true = "requests money back";
    questions["q_refund"] = q3;

    std::cout << "State:\n  \"" << state << "\"\n\n";
    std::cout << "Running decision evaluation across " << questions.size() << " typed questions...\n";

    auto t0 = ncnn::get_current_time();
    LayaResult result = laya.system_one(state, questions);
    auto t1 = ncnn::get_current_time();

    double total_ms = t1 - t0;
    std::cout << "\n========================================================================\n";
    std::cout << "                           Decision Answers                             \n";
    std::cout << "========================================================================\n";

    for (const auto& pair : result.answers) {
        const std::string& qid = pair.first;
        const LayaAnswer& ans = pair.second;

        std::cout << ">>> [" << qid << "] (type: " << ans.type << ")\n";

        if (ans.type == "choice") {
            std::cout << "  Selected Choice:   " << ans.choice << "\n";
            std::cout << "  Probabilities:\n";
            for (const auto& prob : ans.probabilities) {
                std::cout << "    - " << std::left << std::setw(12) << prob.first
                          << ": " << std::fixed << std::setprecision(4) << prob.second << "\n";
            }
        } else if (ans.type == "score") {
            std::cout << "  Expected Score:    " << std::fixed << std::setprecision(4) << ans.score << "\n";
            std::cout << "  Probabilities:\n";
            for (const auto& prob : ans.probabilities) {
                std::string legend_desc = "";
                for (const auto& l : ans.legend) {
                    if (l.first == prob.first) legend_desc = l.second;
                }
                std::cout << "    - Level " << prob.first << " (" << std::left << std::setw(8) << legend_desc << ")"
                          << ": " << std::fixed << std::setprecision(4) << prob.second << "\n";
            }
        } else if (ans.type == "noul") {
            std::cout << "  Noul (P(True)):    " << std::fixed << std::setprecision(4) << ans.noul << "\n";
            std::cout << "  Probabilities:\n";
            for (const auto& prob : ans.probabilities) {
                std::cout << "    - " << std::left << std::setw(8) << prob.first
                          << ": " << std::fixed << std::setprecision(4) << prob.second << "\n";
            }
        }

        std::cout << "  Confidence:        " << std::fixed << std::setprecision(4) << ans.confidence << "\n";
        std::cout << "  Act Probability:   " << std::fixed << std::setprecision(4) << ans.act_probability << "\n\n";
    }

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Total Input Tokens: " << result.input_tokens << "\n";
    std::cout << "Total Latency:      " << std::fixed << std::setprecision(2) << total_ms << " ms ("
              << (total_ms / questions.size()) << " ms/question)\n";
    if (total_ms > 0) {
        double tps = (result.input_tokens * 1000.0) / total_ms;
        std::cout << "Throughput:         " << std::fixed << std::setprecision(1) << tps << " tokens/sec\n";
    }
    std::cout << "========================================================================\n";

    return 0;
}

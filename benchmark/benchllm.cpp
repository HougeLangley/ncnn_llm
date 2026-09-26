#include <float.h>
#include <stdio.h>
#include <string.h>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <net.h>
#include <benchmark.h>
#include <cpu.h>
#include <datareader.h>
#include <layer_type.h>

#include "ncnn_llm_gpt.h"

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* format, void* p) const
    {
        return 0;
    }
    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

static int g_warmup_loop_count = 4;
static int g_loop_count = 4;
static bool g_enable_cooling_down = false;

struct BenchResult
{
    std::string name;
    double prefill_min;
    double prefill_avg;
    double decode_min;
    double decode_avg;
};

BenchResult benchmark_decoder(const char* comment, int hidden_size, int half_embed_dim, int seqlen, const ncnn::Option& opt, const std::string& param_path, bool use_mempool = true)
{
    BenchResult res = {comment, DBL_MAX, 0, DBL_MAX, 0};

    if (!std::filesystem::exists(param_path))
    {
        return res;
    }

    ncnn::Net net;
    net.opt = opt;

    if (net.load_param(param_path.c_str()) != 0)
    {
        fprintf(stderr, "Failed to load param: %s\n", param_path.c_str());
        return res;
    }

    DataReaderFromEmpty dr;
    net.load_model(dr);

    // resolve kv cache blob indexes
    std::vector<int> kv_cache_indexes;
    std::vector<int> out_kv_cache_indexes;
    {
        for (size_t i = 0; i < net.layers().size(); i++)
        {
            const ncnn::Layer* op = net.layers()[i];
            if (op->typeindex != ncnn::LayerType::SDPA)
                continue;

            const size_t input_count = op->bottoms.size();
            const size_t output_count = op->tops.size();

            if (output_count == 3)
            {
                kv_cache_indexes.push_back(op->bottoms[input_count - 2]);
                kv_cache_indexes.push_back(op->bottoms[input_count - 1]);
                out_kv_cache_indexes.push_back(op->tops[output_count - 2]);
                out_kv_cache_indexes.push_back(op->tops[output_count - 1]);
            }
        }
    }

    if (g_enable_cooling_down)
    {
        ncnn::sleep(10 * 1000);
    }

    ncnn::UnlockedPoolAllocator kvcache_allocator;
    kvcache_allocator.set_size_compare_ratio(0.f);

    std::vector<ncnn::Mat> kvcache;

    // prefill
    {
        const int cur_seqlen = seqlen;
        const int past_seqlen = 0;

        ncnn::Mat token_embeds(hidden_size, cur_seqlen);
        token_embeds.fill(0.1f);
        ncnn::Mat attention_mask(past_seqlen + cur_seqlen, cur_seqlen);
        attention_mask.fill(0.f);
        for (int i = 0; i < cur_seqlen; i++)
        {
            float* row = attention_mask.row(i);
            for (int j = past_seqlen + i + 1; j < past_seqlen + cur_seqlen; j++)
            {
                row[j] = -10000.f;
            }
        }
        ncnn::Mat cos_cache(half_embed_dim, cur_seqlen);
        cos_cache.fill(1.f);
        ncnn::Mat sin_cache(half_embed_dim, cur_seqlen);
        sin_cache.fill(0.f);

        std::vector<ncnn::Mat> out_kvcache;
        ncnn::Mat output_states;

        // warm up
        for (int i = 0; i < g_warmup_loop_count; i++)
        {
            ncnn::Extractor ex = net.create_extractor();
            if (use_mempool)
            {
                ex.set_kvcache_allocator(&kvcache_allocator);
                ex.set_kvcache_max_seqlen_hint(seqlen + 128);
            }
            ex.input("in0", token_embeds);
            ex.input("in1", attention_mask);
            ex.input("in2", cos_cache);
            ex.input("in3", sin_cache);

            out_kvcache.resize(out_kv_cache_indexes.size());
            for (size_t k = 0; k < out_kv_cache_indexes.size(); k++)
            {
                ex.extract(out_kv_cache_indexes[k], out_kvcache[k], 1);
            }
            ex.extract("out0", output_states);
        }

        double time_min = DBL_MAX;
        double time_avg = 0;

        for (int i = 0; i < g_loop_count; i++)
        {
            double start = ncnn::get_current_time();
            {
                ncnn::Extractor ex = net.create_extractor();
                if (use_mempool)
                {
                    ex.set_kvcache_allocator(&kvcache_allocator);
                    ex.set_kvcache_max_seqlen_hint(seqlen + 128);
                }
                ex.input("in0", token_embeds);
                ex.input("in1", attention_mask);
                ex.input("in2", cos_cache);
                ex.input("in3", sin_cache);

                out_kvcache.resize(out_kv_cache_indexes.size());
                for (size_t k = 0; k < out_kv_cache_indexes.size(); k++)
                {
                    ex.extract(out_kv_cache_indexes[k], out_kvcache[k], 1);
                }
                ex.extract("out0", output_states);
            }
            double end = ncnn::get_current_time();
            double time = end - start;

            time_min = std::min(time_min, time);
            time_avg += time;
        }

        time_avg /= g_loop_count;
        res.prefill_min = time_min;
        res.prefill_avg = time_avg;
        kvcache = out_kvcache;
    }

    // decode step
    {
        const int cur_seqlen = 1;
        const int past_seqlen = seqlen;

        ncnn::Mat token_embeds(hidden_size, cur_seqlen);
        token_embeds.fill(0.1f);
        ncnn::Mat attention_mask(past_seqlen + cur_seqlen, cur_seqlen);
        attention_mask.fill(0.f);
        ncnn::Mat cos_cache(half_embed_dim, cur_seqlen);
        cos_cache.fill(1.f);
        ncnn::Mat sin_cache(half_embed_dim, cur_seqlen);
        sin_cache.fill(0.f);

        std::vector<ncnn::Mat> out_kvcache;
        ncnn::Mat output_states;

        // warm up
        for (int i = 0; i < g_warmup_loop_count; i++)
        {
            const int cur_past = kvcache.empty() ? past_seqlen : kvcache[0].h;
            ncnn::Mat cur_mask(cur_past + cur_seqlen, cur_seqlen);
            cur_mask.fill(0.f);

            ncnn::Extractor ex = net.create_extractor();
            if (use_mempool)
            {
                ex.set_kvcache_allocator(&kvcache_allocator);
                ex.set_kvcache_max_seqlen_hint(seqlen + 128);
            }
            ex.input("in0", token_embeds);
            ex.input("in1", cur_mask);
            ex.input("in2", cos_cache);
            ex.input("in3", sin_cache);

            for (size_t k = 0; k < kv_cache_indexes.size(); k++)
            {
                ex.input(kv_cache_indexes[k], kvcache[k]);
                if (use_mempool)
                {
                    kvcache[k].release();
                }
            }

            out_kvcache.resize(out_kv_cache_indexes.size());
            for (size_t k = 0; k < out_kv_cache_indexes.size(); k++)
            {
                ex.extract(out_kv_cache_indexes[k], out_kvcache[k], 1);
            }
            ex.extract("out0", output_states);
            kvcache = out_kvcache;
        }

        double time_min = DBL_MAX;
        double time_avg = 0;

        for (int i = 0; i < g_loop_count; i++)
        {
            const int cur_past = kvcache.empty() ? past_seqlen : kvcache[0].h;
            ncnn::Mat cur_mask(cur_past + cur_seqlen, cur_seqlen);
            cur_mask.fill(0.f);

            double start = ncnn::get_current_time();
            {
                ncnn::Extractor ex = net.create_extractor();
                if (use_mempool)
                {
                    ex.set_kvcache_allocator(&kvcache_allocator);
                    ex.set_kvcache_max_seqlen_hint(seqlen + 128);
                }
                ex.input("in0", token_embeds);
                ex.input("in1", cur_mask);
                ex.input("in2", cos_cache);
                ex.input("in3", sin_cache);

                for (size_t k = 0; k < kv_cache_indexes.size(); k++)
                {
                    ex.input(kv_cache_indexes[k], kvcache[k]);
                    if (use_mempool)
                    {
                        kvcache[k].release();
                    }
                }

                out_kvcache.resize(out_kv_cache_indexes.size());
                for (size_t k = 0; k < out_kv_cache_indexes.size(); k++)
                {
                    ex.extract(out_kv_cache_indexes[k], out_kvcache[k], 1);
                }
                ex.extract("out0", output_states);
                kvcache = out_kvcache;
            }
            double end = ncnn::get_current_time();
            double time = end - start;

            time_min = std::min(time_min, time);
            time_avg += time;
        }

        time_avg /= g_loop_count;
        res.decode_min = time_min;
        res.decode_avg = time_avg;
    }

    fprintf(stderr, "%30s (prefill)  min = %7.2f ms  avg = %7.2f ms\n", comment, res.prefill_min, res.prefill_avg);
    fprintf(stderr, "%30s  (decode)  min = %7.2f ms  avg = %7.2f ms\n", comment, res.decode_min, res.decode_avg);

    return res;
}

struct E2EResult
{
    double prefill_ms;
    int token_count;
    double decode_ms;
    double decode_tps;
    double ms_per_token;
};

E2EResult benchmark_e2e_qwen3_single(int num_threads, const std::string& decoder_param_name)
{
    E2EResult res = {0, 0, 0, 0, 0};
    std::string model_dir = "../qwen3_0.6b";
    if (!std::filesystem::exists(model_dir + "/model.json"))
    {
        model_dir = "assets/qwen3_0.6b";
        if (!std::filesystem::exists(model_dir + "/model.json"))
            return res;
    }

    std::string model_json_path = model_dir + "/model.json";
    json config;
    {
        std::ifstream ifs(model_json_path);
        ifs >> config;
    }
    std::string orig_decoder_param = config["params"]["decoder_param"].get<std::string>();

    config["params"]["decoder_param"] = decoder_param_name;
    {
        std::ofstream ofs(model_json_path);
        ofs << config.dump(2);
    }

    {
        ncnn_llm_gpt model(model_dir, false, num_threads);

        std::string prompt = "Please write a short poem about summer.";
        double t0 = ncnn::get_current_time();
        auto ctx = model.prefill(prompt);
        double t1 = ncnn::get_current_time();
        res.prefill_ms = t1 - t0;

        GenerateConfig cfg;
        cfg.max_new_tokens = 32;
        int token_count = 0;
        std::string output_text;

        double t2 = ncnn::get_current_time();
        model.generate(ctx, cfg, [&](const std::string& token) {
            token_count++;
            output_text += token;
        });
        double t3 = ncnn::get_current_time();
        res.decode_ms = t3 - t2;
        res.token_count = token_count;
        res.decode_tps = (token_count > 0 && res.decode_ms > 0) ? (token_count * 1000.0 / res.decode_ms) : 0.0;
        res.ms_per_token = token_count > 0 ? (res.decode_ms / token_count) : 0.0;
    }

    // restore model.json
    config["params"]["decoder_param"] = orig_decoder_param;
    {
        std::ofstream ofs(model_json_path);
        ofs << config.dump(2);
    }

    return res;
}

int main(int argc, char** argv)
{
    int loop_count = 4;
    int num_threads = ncnn::get_physical_big_cpu_count();
    int powersave = 2;
    int gpu_device = -1;
    int cooling_down = 0;
    int seqlen = 233;

    if (argc >= 2) loop_count = atoi(argv[1]);
    if (argc >= 3) num_threads = atoi(argv[2]);
    if (argc >= 4) powersave = atoi(argv[3]);
    if (argc >= 5) gpu_device = atoi(argv[4]);
    if (argc >= 6) cooling_down = atoi(argv[5]);
    if (argc >= 7) seqlen = atoi(argv[6]);

    bool use_vulkan_compute = gpu_device != -1;
    g_enable_cooling_down = cooling_down != 0;
    g_loop_count = loop_count;

    ncnn::set_cpu_powersave(powersave);
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    ncnn::Option opt;
    opt.num_threads = num_threads;
    opt.use_vulkan_compute = use_vulkan_compute;

    fprintf(stderr, "========================================================================\n");
    fprintf(stderr, " ncnn LLM Performance Benchmark: PR #6923 GQA & KVCache Memory Pool    \n");
    fprintf(stderr, "========================================================================\n");
    fprintf(stderr, "Threads = %d, Seqlen = %d, Loop = %d\n\n", num_threads, seqlen, g_loop_count);

    fprintf(stderr, "--- 1. Decoder Subgraph Benchmark (minicpm4_0.5b) ---\n");
    bool has_mcpm_old = std::filesystem::exists("minicpm4_decoder_old.ncnn.param");
    BenchResult mcpm_before  = has_mcpm_old ? benchmark_decoder("minicpm4 (Before PR6923)", 1024, 32, seqlen, opt, "minicpm4_decoder_old.ncnn.param", false) : BenchResult{};
    BenchResult mcpm_gqa     = benchmark_decoder("minicpm4 (After PR6923 NoPool)", 1024, 32, seqlen, opt, "minicpm4_decoder.ncnn.param", false);
    BenchResult mcpm_pool    = benchmark_decoder("minicpm4 (After PR6923 + MemPool)", 1024, 32, seqlen, opt, "minicpm4_decoder.ncnn.param", true);

    fprintf(stderr, "\n--- 2. Decoder Subgraph Benchmark (qwen3_0.6b) ---\n");
    bool has_qwen_old = std::filesystem::exists("../qwen3_0.6b/qwen3_decoder_old.ncnn.param");
    BenchResult qwen_before  = has_qwen_old ? benchmark_decoder("qwen3 (Before PR6923)", 1024, 64, seqlen, opt, "../qwen3_0.6b/qwen3_decoder_old.ncnn.param", false) : BenchResult{};
    BenchResult qwen_gqa     = benchmark_decoder("qwen3 (After PR6923 NoPool)", 1024, 64, seqlen, opt, "../qwen3_0.6b/qwen3_decoder.ncnn.param", false);
    BenchResult qwen_pool    = benchmark_decoder("qwen3 (After PR6923 + MemPool)", 1024, 64, seqlen, opt, "../qwen3_0.6b/qwen3_decoder.ncnn.param", true);

    fprintf(stderr, "\n--- 3. End-to-End Real Model Generation (qwen3_0.6b) ---\n");
    E2EResult e2e_before = {0, 0, 0, 0, 0};
    if (has_qwen_old)
    {
        fprintf(stderr, "Running baseline generation (Before PR 6923)...\n");
        e2e_before = benchmark_e2e_qwen3_single(num_threads, "qwen3_decoder_old.ncnn.param");
        fprintf(stderr, "Before: Prefill = %.2f ms | Decode = %.2f ms (%d tokens, %.2f tokens/s, %.2f ms/token)\n",
                e2e_before.prefill_ms, e2e_before.decode_ms, e2e_before.token_count, e2e_before.decode_tps, e2e_before.ms_per_token);
    }

    fprintf(stderr, "Running optimized generation (After PR 6923 + KVCache MemPool)...\n");
    E2EResult e2e_after = benchmark_e2e_qwen3_single(num_threads, "qwen3_decoder.ncnn.param");
    fprintf(stderr, "After:  Prefill = %.2f ms | Decode = %.2f ms (%d tokens, %.2f tokens/s, %.2f ms/token)\n",
            e2e_after.prefill_ms, e2e_after.decode_ms, e2e_after.token_count, e2e_after.decode_tps, e2e_after.ms_per_token);

    fprintf(stderr, "\n========================================================================\n");
    fprintf(stderr, "                      SPEED COMPARISON SUMMARY                          \n");
    fprintf(stderr, "========================================================================\n");
    fprintf(stderr, "%-32s | %-13s | %-13s | %-9s\n", "Metric", "Before PR6923", "Optimized", "Speedup");
    fprintf(stderr, "---------------------------------+---------------+---------------+----------\n");
    if (has_mcpm_old)
    {
        fprintf(stderr, "%-32s | %9.2f ms   | %9.2f ms   | %+6.1f%%\n", "minicpm4 Prefill (min)", mcpm_before.prefill_min, mcpm_pool.prefill_min, (mcpm_before.prefill_min - mcpm_pool.prefill_min) / mcpm_before.prefill_min * 100.0);
        fprintf(stderr, "%-32s | %9.2f ms   | %9.2f ms   | %+6.1f%%\n", "minicpm4 Decode (min)", mcpm_before.decode_min, mcpm_pool.decode_min, (mcpm_before.decode_min - mcpm_pool.decode_min) / mcpm_before.decode_min * 100.0);
    }
    if (has_qwen_old)
    {
        fprintf(stderr, "%-32s | %9.2f ms   | %9.2f ms   | %+6.1f%%\n", "qwen3 Prefill (min)", qwen_before.prefill_min, qwen_pool.prefill_min, (qwen_before.prefill_min - qwen_pool.prefill_min) / qwen_before.prefill_min * 100.0);
        fprintf(stderr, "%-32s | %9.2f ms   | %9.2f ms   | %+6.1f%%\n", "qwen3 Decode (NoPool -> Pool)", qwen_gqa.decode_min, qwen_pool.decode_min, (qwen_gqa.decode_min - qwen_pool.decode_min) / qwen_gqa.decode_min * 100.0);
        fprintf(stderr, "%-32s | %9.2f ms   | %9.2f ms   | %+6.1f%%\n", "qwen3 Decode Overall (min)", qwen_before.decode_min, qwen_pool.decode_min, (qwen_before.decode_min - qwen_pool.decode_min) / qwen_before.decode_min * 100.0);
        fprintf(stderr, "%-32s | %9.2f ms   | %9.2f ms   | %+6.1f%%\n", "qwen3 E2E Prefill", e2e_before.prefill_ms, e2e_after.prefill_ms, (e2e_before.prefill_ms - e2e_after.prefill_ms) / e2e_before.prefill_ms * 100.0);
        fprintf(stderr, "%-32s | %9.2f ms   | %9.2f ms   | %+6.1f%%\n", "qwen3 E2E Decode/Token", e2e_before.ms_per_token, e2e_after.ms_per_token, (e2e_before.ms_per_token - e2e_after.ms_per_token) / e2e_before.ms_per_token * 100.0);
        fprintf(stderr, "%-32s | %9.2f tps  | %9.2f tps  | %+6.1f%%\n", "qwen3 E2E Decode Speed", e2e_before.decode_tps, e2e_after.decode_tps, (e2e_after.decode_tps - e2e_before.decode_tps) / e2e_before.decode_tps * 100.0);
    }
    fprintf(stderr, "========================================================================\n");

    return 0;
}

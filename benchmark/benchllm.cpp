// benchmark/benchllm.cpp
//
// Qwen3-0.6B bf16 / int8 end-to-end benchmark, llama.cpp style:
//   pp256 = prefill a prompt of EXACTLY 256 tokens, measure wall time
//   tg64  = autoregressive decode of 64 tokens, measure ms/token
//
// The prompt is built with a mirrored BpeTokenizer (same vocab/merges/byte-encoder
// config as ncnn_llm_gpt) and verified by an encode -> decode -> re-encode fixed
// point check, so the token count fed into prefill is exact, not approximate.
//
// usage: benchllm [loop_count] [threads] [powersave] [gpu_device] [cooling_down] [pp] [tg]
//   Any numeric argument <= 0 (or omitted) falls back to the default.
//   loop_count    measured passes per model (default 4)
//   threads       cpu threads (default 4)
//   powersave     0 = all cores, 1 = little only, 2 = big only (default 2)
//   gpu_device    -1 = CPU (default), >=0 = vulkan device index
//   cooling_down  1 = sleep between passes (default 0)
//   pp            prefill token count (default 256)
//   tg            decode token count (default 64)

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <net.h>
#include <benchmark.h>
#include <cpu.h>

#include "ncnn_llm_gpt.h"
#include "utils/tokenizer/bpe_tokenizer.h"

static const char* PROMPT_SEED =
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! Sphinx of black quartz, judge my vow. "
    "The five boxing wizards jump quickly over the sleepy dog again and again. ";

struct ModelResult
{
    bool ran = false;
    std::string name;
    std::string label;
    std::string dir;
    int pp_tokens = 0;      // verified prefill token count
    int tg_tokens = 0;      // requested decode token count
    double pp_min = DBL_MAX;
    double pp_avg = 0;
    double pp_tps = 0;
    double tg_ms_min = DBL_MAX;   // ms per token
    double tg_ms_avg = 0;         // ms per token
    double tg_tps = 0;
    int tg_actual = 0;      // sum of actually generated tokens across passes
    std::string sample_text;
};

// Resolve a model directory by trying several relative search paths.
static std::string resolve_model_dir(const std::string& name)
{
    const char* search_paths[] = {
        "./assets", ".", "../assets", "../", "../../assets", "../../"
    };
    for (const char* base : search_paths)
    {
        std::string dir = std::string(base) + "/" + name;
        if (std::filesystem::exists(dir + "/model.json"))
            return dir;
    }
    return "";
}

// Load a tokenizer with exactly the same construction as ncnn_llm_gpt,
// so token counts computed here match what prefill() will do internally.
static std::unique_ptr<BpeTokenizer> load_tokenizer(const std::string& dir)
{
    nlohmann::json config;
    {
        std::ifstream ifs(dir + "/model.json");
        ifs >> config;
    }

    std::string type = "bpe";
    if (config["tokenizer"].contains("type"))
        type = config["tokenizer"]["type"].get<std::string>();

    std::string vocab_file = dir + "/" + config["tokenizer"]["vocab_file"].get<std::string>();
    std::string merges_file = dir + "/" + config["tokenizer"]["merges_file"].get<std::string>();

    std::unique_ptr<BpeTokenizer> tok(new BpeTokenizer(BpeTokenizer::LoadFromFiles(
        vocab_file, merges_file, SpecialTokensConfig{}, false, true, type == "bbpe")));

    if (config["tokenizer"].contains("additional_special_tokens"))
    {
        for (const auto& t : config["tokenizer"]["additional_special_tokens"].get<std::vector<std::string>>())
            tok->AddAdditionalSpecialToken(t);
    }
    return tok;
}

// Build a plain-text prompt that encodes to exactly `n` tokens.
static std::string build_prompt_of_tokens(const BpeTokenizer& tok, int n, int& actual_tokens)
{
    std::vector<int> ids = tok.encode(PROMPT_SEED, false, false);
    while ((int)ids.size() < n)
    {
        std::vector<int> more = tok.encode(PROMPT_SEED, false, false);
        ids.insert(ids.end(), more.begin(), more.end());
    }
    if ((int)ids.size() > n)
        ids.resize(n);

    // Byte-level BPE round-trips exactly, so the first iteration is normally
    // already a fixed point. Iterate a few times just in case.
    for (int it = 0; it < 8; it++)
    {
        std::string text = tok.decode(ids, false);
        std::vector<int> re = tok.encode(text, false, false);
        if ((int)re.size() == n)
        {
            actual_tokens = n;
            return text;
        }

        std::vector<int> next = re;
        if ((int)next.size() > n)
            next.resize(n);
        while ((int)next.size() < n)
        {
            std::vector<int> more = tok.encode(PROMPT_SEED, false, false);
            next.insert(next.end(), more.begin(), more.end());
        }
        if ((int)next.size() > n)
            next.resize(n);
        ids = next;
    }

    // Fallback: decode without fixed-point guarantee, report the real count.
    std::string text = tok.decode(ids, false);
    actual_tokens = (int)tok.encode(text, false, false).size();
    return text;
}

struct PassResult
{
    double prefill_ms;
    int prefill_tokens;
    double decode_ms;
    int decode_tokens;
    std::string text;
};

static PassResult run_pass(ncnn_llm_gpt& model, const std::string& prompt, int pp_tokens, int tg_tokens)
{
    PassResult r = {0, pp_tokens, 0, 0, ""};

    double t0 = ncnn::get_current_time();
    auto ctx = model.prefill(prompt);
    double t1 = ncnn::get_current_time();

    GenerateConfig cfg;
    cfg.max_new_tokens = tg_tokens;
    cfg.temperature = 0.f;
    cfg.do_sample = 0;
    cfg.repetition_penalty = 1.f;

    int tokens = 0;
    std::string text;
    double t2 = ncnn::get_current_time();
    model.generate(ctx, cfg, [&](const std::string& token) {
        tokens++;
        text += token;
    });
    double t3 = ncnn::get_current_time();

    r.prefill_ms = t1 - t0;
    r.decode_ms = t3 - t2;
    r.decode_tokens = tokens;
    r.text = text;
    return r;
}

static std::string one_line(const std::string& s, size_t max_len)
{
    std::string out;
    for (char c : s)
    {
        if (c == '\n' || c == '\r' || c == '\t') out += ' ';
        else out += c;
        if (out.size() >= max_len) break;
    }
    return out;
}

int main(int argc, char** argv)
{
    // Sentinels: 0 or negative for any numeric argument = use the default.
    int loop_count = 0;
    int num_threads = 0;
    int powersave = 2;
    int gpu_device = -1;
    int cooling_down = 0;
    int pp_tokens = 0;
    int tg_tokens = 0;

    if (argc >= 2) loop_count = atoi(argv[1]);
    if (argc >= 3) num_threads = atoi(argv[2]);
    if (argc >= 4) powersave = atoi(argv[3]);
    if (argc >= 5) gpu_device = atoi(argv[4]);
    if (argc >= 6) cooling_down = atoi(argv[5]);
    if (argc >= 7) pp_tokens = atoi(argv[6]);
    if (argc >= 8) tg_tokens = atoi(argv[7]);

    // Sanitize: never trust a non-positive value.
    if (loop_count <= 0) loop_count = 4;
    if (num_threads <= 0) num_threads = 4;      // fixed default per user request
    if (powersave < 0 || powersave > 2) powersave = 2;
    if (gpu_device < -1) gpu_device = -1;
    if (cooling_down != 0 && cooling_down != 1) cooling_down = 0;
    if (pp_tokens <= 0) pp_tokens = 256;
    if (tg_tokens <= 0) tg_tokens = 64;

    bool use_vulkan = gpu_device != -1;

    ncnn::set_cpu_powersave(powersave);
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    fprintf(stderr, "========================================================================\n");
    fprintf(stderr, "  ncnn LLM Benchmark: Qwen3-0.6B bf16 vs int8 (E2E pp/tg)               \n");
    fprintf(stderr, "========================================================================\n");
    fprintf(stderr, "Threads = %d, PP = %d, TG = %d, Loop = %d, Vulkan = %s\n\n",
            num_threads, pp_tokens, tg_tokens, loop_count, use_vulkan ? "on" : "off");

    struct ModelSpec { const char* name; const char* label; };
    const ModelSpec specs[] = {
        { "qwen3_0.6b", "bf16" },
        { "qwen3_0.6b_int8", "int8" },
    };

    std::vector<ModelResult> results;

    for (const ModelSpec& spec : specs)
    {
        ModelResult res;
        res.name = spec.name;
        res.label = spec.label;
        res.tg_tokens = tg_tokens;

        std::string dir = resolve_model_dir(spec.name);
        if (dir.empty())
        {
            fprintf(stderr, "--- %s (%s) ---\n  SKIP: model dir not found\n\n", spec.name, spec.label);
            results.push_back(res);
            continue;
        }
        res.dir = dir;

        fprintf(stderr, "--- %s (%s) ---\n  model dir: %s\n", spec.name, spec.label, dir.c_str());

        // Build the exact pp-token prompt.
        int prompt_tokens = 0;
        std::string prompt;
        {
            std::unique_ptr<BpeTokenizer> tok = load_tokenizer(dir);
            prompt = build_prompt_of_tokens(*tok, pp_tokens, prompt_tokens);
        }
        res.pp_tokens = prompt_tokens;
        fprintf(stderr, "  prompt: \"%s...\" (%d tokens%s)\n",
                one_line(prompt, 48).c_str(), prompt_tokens,
                prompt_tokens == pp_tokens ? ", exact" : ", WARN: not exact");

        // use_bf16 = true matches the default runtime config (llm_ncnn_run);
        // the int8 model's block-quantized weights are dispatched by the param anyway.
        ncnn_llm_gpt model(dir, use_vulkan, num_threads, use_vulkan ? gpu_device : 0, true);

        // warmup pass (untimed)
        run_pass(model, prompt, prompt_tokens, tg_tokens);

        double pp_sum = 0;
        int tg_sum = 0;
        double tg_ms_sum = 0;
        for (int i = 0; i < loop_count; i++)
        {
            if (cooling_down)
                ncnn::sleep(10 * 1000);

            PassResult pr = run_pass(model, prompt, prompt_tokens, tg_tokens);

            res.pp_min = std::min(res.pp_min, pr.prefill_ms);
            pp_sum += pr.prefill_ms;
            res.tg_actual += pr.decode_tokens;
            tg_sum += pr.decode_tokens;
            tg_ms_sum += pr.decode_ms;

            if (pr.decode_tokens > 0)
                res.tg_ms_min = std::min(res.tg_ms_min, pr.decode_ms / pr.decode_tokens);

            if (i == 0)
                res.sample_text = pr.text;

            fprintf(stderr, "  pass %d: pp = %7.2f ms   tg = %7.2f ms (%d/%d tok, %6.2f ms/tok)\n",
                    i + 1, pr.prefill_ms, pr.decode_ms, pr.decode_tokens, tg_tokens,
                    pr.decode_tokens > 0 ? pr.decode_ms / pr.decode_tokens : 0.0);

            if (pr.decode_tokens < tg_tokens)
                fprintf(stderr, "  WARN: decode stopped early (EOS after %d tokens)\n", pr.decode_tokens);
        }

        res.pp_avg = pp_sum / loop_count;
        res.pp_tps = pp_sum > 0 ? (double)prompt_tokens * loop_count * 1000.0 / pp_sum : 0;
        res.tg_ms_avg = tg_sum > 0 ? tg_ms_sum / tg_sum : 0;
        res.tg_tps = tg_ms_sum > 0 ? tg_sum * 1000.0 / tg_ms_sum : 0;
        res.ran = true;

        fprintf(stderr, "  SUMMARY pp%d: min = %7.2f ms  avg = %7.2f ms  (%.1f tokens/s)\n",
                pp_tokens, res.pp_min, res.pp_avg, res.pp_tps);
        fprintf(stderr, "  SUMMARY tg%d: min = %6.2f ms/tok  avg = %6.2f ms/tok  (%.1f tokens/s)\n",
                tg_tokens, res.tg_ms_min, res.tg_ms_avg, res.tg_tps);
        fprintf(stderr, "  sample: \"%s\"\n\n", one_line(res.sample_text, 72).c_str());

        results.push_back(res);
    }

    // summary table
    fprintf(stderr, "========================================================================\n");
    fprintf(stderr, "                            SPEED SUMMARY                               \n");
    fprintf(stderr, "========================================================================\n");
    fprintf(stderr, "%-22s | %10s | %10s | %12s | %12s\n",
            "Model", "pp min ms", "pp avg ms", "tg ms/tok avg", "tg tokens/s");
    fprintf(stderr, "-----------------------+------------+------------+---------------+-------------\n");
    for (const ModelResult& r : results)
    {
        if (!r.ran)
        {
            fprintf(stderr, "%-22s | %10s | %10s | %12s | %12s\n",
                    (r.name + " (" + r.label + ")").c_str(), "-", "-", "-", "-");
            continue;
        }
        fprintf(stderr, "%-22s | %10.2f | %10.2f | %12.2f | %12.1f\n",
                (r.name + " (" + r.label + ")").c_str(),
                r.pp_min, r.pp_avg, r.tg_ms_avg, r.tg_tps);
    }

    const ModelResult* bf16 = nullptr;
    const ModelResult* int8 = nullptr;
    for (const ModelResult& r : results)
    {
        if (r.label == "bf16" && r.ran) bf16 = &r;
        if (r.label == "int8" && r.ran) int8 = &r;
    }
    if (bf16 && int8 && bf16->pp_avg > 0 && int8->pp_avg > 0 && bf16->tg_ms_avg > 0 && int8->tg_ms_avg > 0)
    {
        fprintf(stderr, "-----------------------+------------+------------+---------------+-------------\n");
        fprintf(stderr, " int8 vs bf16          | %9.1f%% | %9.1f%% | %11.1f%% | %11.1f%%\n",
                (bf16->pp_min - int8->pp_min) / bf16->pp_min * 100.0,
                (bf16->pp_avg - int8->pp_avg) / bf16->pp_avg * 100.0,
                (bf16->tg_ms_avg - int8->tg_ms_avg) / bf16->tg_ms_avg * 100.0,
                (int8->tg_tps - bf16->tg_tps) / bf16->tg_tps * 100.0);
        fprintf(stderr, " (positive = int8 faster)\n");
    }
    fprintf(stderr, "========================================================================\n");

    return 0;
}

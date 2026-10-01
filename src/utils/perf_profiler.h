#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>

#include <benchmark.h>

namespace ncnn_llm {

struct PhaseStat {
    std::string name;
    int count = 0;
    double total_ms = 0.0;
    double min_ms = 1e9;
    double max_ms = 0.0;

    PhaseStat() = default;
    explicit PhaseStat(std::string n) : name(std::move(n)) {}

    void record(double ms) {
        if (count == 0) {
            min_ms = ms;
            max_ms = ms;
        } else {
            if (ms < min_ms) min_ms = ms;
            if (ms > max_ms) max_ms = ms;
        }
        count++;
        total_ms += ms;
    }

    void accumulate(const PhaseStat& other) {
        if (other.count == 0) return;
        if (count == 0) {
            min_ms = other.min_ms;
            max_ms = other.max_ms;
        } else {
            if (other.min_ms < min_ms) min_ms = other.min_ms;
            if (other.max_ms > max_ms) max_ms = other.max_ms;
        }
        count += other.count;
        total_ms += other.total_ms;
    }

    double avg_ms() const {
        return count > 0 ? (total_ms / count) : 0.0;
    }

    void reset() {
        count = 0;
        total_ms = 0.0;
        min_ms = 1e9;
        max_ms = 0.0;
    }
};

class ScopedTimer {
public:
    explicit ScopedTimer(PhaseStat& stat)
        : stat_(&stat), start_(std::chrono::steady_clock::now()) {}

    explicit ScopedTimer(PhaseStat* stat)
        : stat_(stat), start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        if (stat_) {
            auto end = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start_).count();
            stat_->record(ms);
        }
    }

private:
    PhaseStat* stat_ = nullptr;
    std::chrono::steady_clock::time_point start_;
};

struct PrefillPerfStats {
    int prompt_tokens = 0;
    double total_ms = 0.0;

    PhaseStat tokenizer_encode{"Tokenizer Encode"};
    PhaseStat vision_feature_extract{"Vision Feature Extract"};
    PhaseStat rope_cache{"RoPE Cache Gen"};
    PhaseStat embed_lookup{"Token Embed Lookup"};
    PhaseStat causal_mask{"Causal Mask Gen"};
    PhaseStat decoder_prompt{"Decoder Prompt Eval"};
    PhaseStat kv_cache_extract{"KV Cache Extract"};
    PhaseStat recurrent_cache_extract{"Recurrent State Extract"};
    PhaseStat last_token_embed{"Last Token Embed"};
    PhaseStat last_token_rope{"Last Token RoPE"};
    PhaseStat last_token_decoder{"Last Token Decoder"};
    PhaseStat lm_head{"LM Head (Proj Out)"};
    PhaseStat sampling{"Sampling (Argmax)"};

    std::vector<const PhaseStat*> all_phases() const {
        return {
            &tokenizer_encode,
            &vision_feature_extract,
            &rope_cache,
            &embed_lookup,
            &causal_mask,
            &decoder_prompt,
            &kv_cache_extract,
            &recurrent_cache_extract,
            &last_token_embed,
            &last_token_rope,
            &last_token_decoder,
            &lm_head,
            &sampling
        };
    }

    void accumulate(const PrefillPerfStats& other) {
        prompt_tokens += other.prompt_tokens;
        total_ms += other.total_ms;
        tokenizer_encode.accumulate(other.tokenizer_encode);
        vision_feature_extract.accumulate(other.vision_feature_extract);
        rope_cache.accumulate(other.rope_cache);
        embed_lookup.accumulate(other.embed_lookup);
        causal_mask.accumulate(other.causal_mask);
        decoder_prompt.accumulate(other.decoder_prompt);
        kv_cache_extract.accumulate(other.kv_cache_extract);
        recurrent_cache_extract.accumulate(other.recurrent_cache_extract);
        last_token_embed.accumulate(other.last_token_embed);
        last_token_rope.accumulate(other.last_token_rope);
        last_token_decoder.accumulate(other.last_token_decoder);
        lm_head.accumulate(other.lm_head);
        sampling.accumulate(other.sampling);
    }

    double tokens_per_second() const {
        return total_ms > 0 ? (prompt_tokens * 1000.0 / total_ms) : 0.0;
    }

    void reset() {
        prompt_tokens = 0;
        total_ms = 0.0;
        tokenizer_encode.reset();
        vision_feature_extract.reset();
        rope_cache.reset();
        embed_lookup.reset();
        causal_mask.reset();
        decoder_prompt.reset();
        kv_cache_extract.reset();
        recurrent_cache_extract.reset();
        last_token_embed.reset();
        last_token_rope.reset();
        last_token_decoder.reset();
        lm_head.reset();
        sampling.reset();
    }
};

struct DecodePerfStats {
    int decode_tokens = 0;
    double total_ms = 0.0;

    PhaseStat embed_lookup{"Token Embed Lookup"};
    PhaseStat rope_gen{"RoPE Embed Gen"};
    PhaseStat mask_gen{"Mask Gen"};
    PhaseStat decoder_step{"Decoder Step"};
    PhaseStat kv_cache_update{"KV Cache Update"};
    PhaseStat recurrent_cache_update{"Recurrent State Update"};
    PhaseStat lm_head{"LM Head (Proj Out)"};
    PhaseStat sampling{"Sampling"};
    PhaseStat tokenizer_decode{"Tokenizer Decode"};
    PhaseStat callback{"Callback"};

    std::vector<const PhaseStat*> all_phases() const {
        return {
            &embed_lookup,
            &rope_gen,
            &mask_gen,
            &decoder_step,
            &kv_cache_update,
            &recurrent_cache_update,
            &lm_head,
            &sampling,
            &tokenizer_decode,
            &callback
        };
    }

    double tokens_per_second() const {
        return total_ms > 0 ? (decode_tokens * 1000.0 / total_ms) : 0.0;
    }

    double ms_per_token() const {
        return decode_tokens > 0 ? (total_ms / decode_tokens) : 0.0;
    }

    void reset() {
        decode_tokens = 0;
        total_ms = 0.0;
        embed_lookup.reset();
        rope_gen.reset();
        mask_gen.reset();
        decoder_step.reset();
        kv_cache_update.reset();
        recurrent_cache_update.reset();
        lm_head.reset();
        sampling.reset();
        tokenizer_decode.reset();
        callback.reset();
    }
};

struct ModelMetadata {
    std::string model_path;
    std::string model_type;
    int num_threads = 0;
    bool use_vulkan = false;
    int vulkan_device = 0;
    bool use_bf16 = false;
};

class LlmPerfReport {
public:
    ModelMetadata meta;
    PrefillPerfStats prefill;
    DecodePerfStats decode;

    bool has_layer_stats = false;
#if NCNN_BENCHMARK
    std::vector<ncnn::LayerTypeStat> layer_type_stats;
    std::vector<ncnn::LayerBenchStat> layer_stats;
#endif

    std::string to_string(bool include_layer_stats = true) const;
    void print(bool include_layer_stats = true) const;
    nlohmann::json to_json(bool include_layer_stats = true) const;
    std::string analyze_bottlenecks() const;
};

void set_perf_level(int level);
int get_perf_level();
bool is_perf_enabled();

} // namespace ncnn_llm

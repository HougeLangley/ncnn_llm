#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mat.h>
#include <net.h>

#include "ncnn_llm_base.h"

struct LlmTokenSampleConfig {
    int vocab_size = 0;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    float repetition_penalty = 1.0f;
    int do_sample = 0;
};

ncnn::Mat llm_run_text_embed(ncnn::Net& embed_net, const std::vector<int>& input_ids);
ncnn::Mat llm_run_text_embed(ncnn::Net& embed_net, int token_id);

ncnn::Mat llm_run_decoder_with_kv(ncnn::Net& decoder_net,
                                  const ncnn::Mat& embeds,
                                  const ncnn::Mat& mask,
                                  const ncnn::Mat& cos_cache,
                                  const ncnn::Mat& sin_cache,
                                  KVCache& kv_cache,
                                  int attn_cnt,
                                  bool is_prefill,
                                  ncnn::Allocator* kvcache_allocator = nullptr,
                                  int max_seqlen_hint = 0);

ncnn::Mat llm_run_lm_head(ncnn::Net& lm_head_net, const ncnn::Mat& hidden_states);
namespace ncnn_llm {
class LlmHead;
}

ncnn::Mat llm_run_lm_head(const ncnn_llm::LlmHead& lm_head, const ncnn::Mat& hidden_states, const ncnn::Option& opt);

int llm_select_next_token(const ncnn::Mat& logits,
                          const std::unordered_map<int, int>& history_counts,
                          const LlmTokenSampleConfig& cfg,
                          const std::vector<int>* generated_tokens = nullptr);

int llm_select_next_token(const ncnn::Mat& logits,
                          const std::unordered_set<int>& history,
                          const LlmTokenSampleConfig& cfg);

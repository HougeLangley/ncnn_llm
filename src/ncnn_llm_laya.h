#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <utility>

#include <mat.h>
#include <net.h>
#include <nlohmann/json.hpp>

#include "ncnn_llm_base.h"
#include "utils/tokenizer/bpe_tokenizer.h"
#include "utils/rope_embed.h"

// Question representation for Laya decision engine
struct LayaQuestion {
    std::string type; // "choice", "score", "noul"
    std::string instructions;
    // For "choice": label -> description
    std::vector<std::pair<std::string, std::string>> choice_criteria;
    // For "score": level descriptions
    std::vector<std::string> score_criteria;
    // For "noul": false/true descriptions (optional)
    std::string noul_false;
    std::string noul_true;
};

// Answer representation
struct LayaAnswer {
    std::string type;
    std::string choice; // for choice: selected label
    float score = 0.0f; // for score: expected score
    float noul = 0.0f;  // for noul: probability of True [0, 1]
    std::vector<std::pair<std::string, float>> probabilities; // label -> calibrated prob
    std::vector<std::pair<std::string, std::string>> legend; // for score: index -> criteria
    float confidence = 0.0f;      // mathematical confidence: 1 - H/log(k)
    float act_probability = 0.0f; // RL agent escalation probability
};

struct LayaResult {
    std::string model = "rl-agent";
    std::map<std::string, LayaAnswer> answers;
    int input_tokens = 0;
};

class ncnn_llm_laya : public ncnn_llm_base {
public:
    ncnn_llm_laya(const std::string& model_path,
                  bool use_vulkan = false,
                  int num_threads = 4,
                  int vulkan_device = 0,
                  bool use_bf16 = true);

    // Full system_one inference (Jev-compatible request format)
    LayaResult system_one(const std::string& state,
                          const std::map<std::string, LayaQuestion>& questions) const;

    // JSON-based inference (accepts Jev / Python dict format, returns standard result JSON)
    nlohmann::json system_one_json(const std::string& state,
                                   const nlohmann::json& questions_json) const;

    // Single question convenience methods
    LayaAnswer decide_choice(const std::string& state,
                             const std::string& instructions,
                             const std::vector<std::pair<std::string, std::string>>& criteria) const;

    LayaAnswer decide_choice(const std::string& state,
                             const std::string& instructions,
                             const std::vector<std::string>& options) const;

    LayaAnswer decide_score(const std::string& state,
                            const std::string& instructions,
                            const std::vector<std::string>& criteria) const;

    LayaAnswer decide_noul(const std::string& state,
                           const std::string& instructions,
                           const std::string& false_crit = "",
                           const std::string& true_crit = "") const;

    bool ok() const { return ok_; }
    const std::string& model_type() const { return model_type_; }
    int hidden_size() const { return hidden_size_; }

private:
    std::shared_ptr<ncnn::Net> backbone_net_;
    std::shared_ptr<ncnn::Net> scorer_net_;
    std::shared_ptr<ncnn::Net> act_head_net_;
    std::shared_ptr<BpeTokenizer> tokenizer_;

    std::string model_type_ = "laya";
    std::string mask_str_ = "[MASK]";
    int hidden_size_ = 1024;
    int head_dim_ = 64;
    int num_heads_ = 16;
    int max_len_ = 512;
    int head_max_len_ = 192;
    float rope_theta_full_ = 160000.0f;
    float rope_theta_slide_ = 10000.0f;
    int sliding_window_ = 128;

    std::vector<float> temperature_ = {1.6369f, 1.2514f, 1.9834f};
    std::map<std::string, float> temperature_by_options_;

    // Internal question representation
    struct InternalQ {
        std::string t; // "choice", "score", "noul"
        std::string ins;
        std::vector<std::string> options_render;
        std::vector<std::string> keys;
        std::vector<std::string> crit_descriptions;
    };

    InternalQ parse_question(const LayaQuestion& q) const;
    InternalQ parse_question_json(const nlohmann::json& qdef) const;

    void build_sequence(const std::string& state,
                        const InternalQ& q,
                        std::vector<int>& out_ids,
                        std::vector<int>& out_markers) const;


    void build_sliding_mask(int seq_len, int window, ncnn::Mat& mask_mat) const;

    std::string temp_bucket(int qtype, int k) const;
    float confidence_from_probs(const std::vector<float>& p, int k) const;

    LayaAnswer evaluate_single_question(const std::string& state,
                                       const InternalQ& q,
                                       int& token_count) const;
};

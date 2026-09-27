// Copyright (c) 2026 ncnn_llm authors. All rights reserved.
// Use of this source code is governed by a BSD-style license.

#include "ncnn_llm_laya.h"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

static void replace_all(std::string& str, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
}

ncnn_llm_laya::ncnn_llm_laya(const std::string& model_path,
                             bool use_vulkan,
                             int num_threads,
                             int vulkan_device,
                             bool use_bf16)
    : ncnn_llm_base(use_vulkan, num_threads > 0 ? num_threads : 4) {
    try {
        nlohmann::json config;
        {
            std::ifstream ifs(model_path + "/model.json");
            if (!ifs.is_open()) {
                fprintf(stderr, "[ncnn_llm_laya] cannot open %s/model.json\n", model_path.c_str());
                ok_ = false;
                return;
            }
            ifs >> config;
        }

        if (config.contains("model_type")) {
            model_type_ = config["model_type"].get<std::string>();
        }

        backbone_net_ = std::make_shared<ncnn::Net>();
        scorer_net_ = std::make_shared<ncnn::Net>();
        act_head_net_ = std::make_shared<ncnn::Net>();

        auto configure_opt = [&](std::shared_ptr<ncnn::Net>& net) {
            if (num_threads > 0) {
                net->opt.num_threads = num_threads;
            }
            if (use_vulkan) {
#if NCNN_VULKAN
                net->opt.use_vulkan_compute = true;
                net->opt.vulkan_device_index = vulkan_device >= 0 ? vulkan_device : 0;
#endif
            }
            if (use_bf16) {
                net->opt.use_bf16_storage = true;
            }
        };

        configure_opt(backbone_net_);
        configure_opt(scorer_net_);
        configure_opt(act_head_net_);

        auto p = [&](const char* key) {
            return model_path + "/" + config["params"][key].get<std::string>();
        };

        std::string bb_param = p("backbone_param");
        std::string bb_bin = p("backbone_bin");
        std::string sc_param = p("scorer_param");
        std::string sc_bin = p("scorer_bin");
        std::string act_param = p("act_head_param");
        std::string act_bin = p("act_head_bin");

        printf("[ncnn_llm_laya] Loading Laya model from %s\n", model_path.c_str());
        printf("  Backbone: %s\n", bb_param.c_str());
        printf("  Scorer:   %s\n", sc_param.c_str());
        printf("  ActHead:  %s\n", act_param.c_str());

        if (backbone_net_->load_param(bb_param.c_str()) != 0 ||
            backbone_net_->load_model(bb_bin.c_str()) != 0) {
            fprintf(stderr, "[ncnn_llm_laya] Failed to load backbone model!\n");
            ok_ = false;
            return;
        }

        if (scorer_net_->load_param(sc_param.c_str()) != 0 ||
            scorer_net_->load_model(sc_bin.c_str()) != 0) {
            fprintf(stderr, "[ncnn_llm_laya] Failed to load scorer model!\n");
            ok_ = false;
            return;
        }

        if (act_head_net_->load_param(act_param.c_str()) != 0 ||
            act_head_net_->load_model(act_bin.c_str()) != 0) {
            fprintf(stderr, "[ncnn_llm_laya] Failed to load act head model!\n");
            ok_ = false;
            return;
        }

        // Tokenizer
        std::string vocab_file = model_path + "/" + config["tokenizer"]["vocab_file"].get<std::string>();
        std::string merges_file = model_path + "/" + config["tokenizer"]["merges_file"].get<std::string>();

        SpecialTokensConfig spec;
        if (config["tokenizer"].contains("cls")) spec.cls_token = config["tokenizer"]["cls"].get<std::string>();
        if (config["tokenizer"].contains("sep")) spec.sep_token = config["tokenizer"]["sep"].get<std::string>();
        if (config["tokenizer"].contains("pad")) spec.pad_token = config["tokenizer"]["pad"].get<std::string>();
        if (config["tokenizer"].contains("mask")) {
            spec.mask_token = config["tokenizer"]["mask"].get<std::string>();
            mask_str_ = *spec.mask_token;
        }
        if (config["tokenizer"].contains("unk")) spec.unk_token = config["tokenizer"]["unk"].get<std::string>();

        bool use_byte_encoder = true;
        if (config["tokenizer"].contains("use_byte_encoder")) {
            use_byte_encoder = config["tokenizer"]["use_byte_encoder"].get<bool>();
        }
        tokenizer_ = std::make_shared<BpeTokenizer>(BpeTokenizer::LoadFromFiles(
            vocab_file, merges_file, spec, false, true, use_byte_encoder
        ));

        // Settings
        if (config.contains("setting")) {
            auto& s = config["setting"];
            if (s.contains("hidden_size")) hidden_size_ = s["hidden_size"].get<int>();
            if (s.contains("head_dim")) head_dim_ = s["head_dim"].get<int>();
            if (s.contains("num_heads")) num_heads_ = s["num_heads"].get<int>();
            if (s.contains("max_len")) max_len_ = s["max_len"].get<int>();
            if (s.contains("head_max_len")) head_max_len_ = s["head_max_len"].get<int>();
            if (s.contains("rope_theta_full")) rope_theta_full_ = s["rope_theta_full"].get<float>();
            if (s.contains("rope_theta_slide")) rope_theta_slide_ = s["rope_theta_slide"].get<float>();
            if (s.contains("sliding_window")) sliding_window_ = s["sliding_window"].get<int>();
            if (s.contains("temperature")) {
                temperature_.clear();
                for (auto& v : s["temperature"]) temperature_.push_back(v.get<float>());
            }
            if (s.contains("temperature_by_options")) {
                temperature_by_options_.clear();
                for (auto it = s["temperature_by_options"].begin(); it != s["temperature_by_options"].end(); ++it) {
                    temperature_by_options_[it.key()] = it.value().get<float>();
                }
            }
        }

        printf("  Loaded vocab size: %zu\n", tokenizer_->vocab_size());
        printf("  Hidden size: %d, Max len: %d, Head max len: %d\n", hidden_size_, max_len_, head_max_len_);
        ok_ = true;

    } catch (const std::exception& e) {
        fprintf(stderr, "[ncnn_llm_laya] Exception while loading model: %s\n", e.what());
        ok_ = false;
    }
}

ncnn_llm_laya::InternalQ ncnn_llm_laya::parse_question(const LayaQuestion& q) const {
    InternalQ iq;
    iq.t = q.type;
    iq.ins = q.instructions;

    if (q.type == "choice") {
        for (const auto& item : q.choice_criteria) {
            iq.keys.push_back(item.first);
            iq.crit_descriptions.push_back(item.second);
            if (item.second.empty()) {
                iq.options_render.push_back(item.first);
            } else {
                iq.options_render.push_back(item.first + ": " + item.second);
            }
        }
    } else if (q.type == "score") {
        for (size_t i = 0; i < q.score_criteria.size(); ++i) {
            std::string idx_str = std::to_string(i);
            iq.keys.push_back(idx_str);
            iq.crit_descriptions.push_back(q.score_criteria[i]);
            iq.options_render.push_back("level " + idx_str + ": " + q.score_criteria[i]);
        }
    } else { // "noul"
        iq.t = "noul";
        iq.keys = {"false", "true"};
        std::string f_crit = q.noul_false.empty() ? "no, the statement does not hold" : q.noul_false;
        std::string t_crit = q.noul_true.empty() ? "yes, the statement holds" : q.noul_true;
        iq.crit_descriptions = {f_crit, t_crit};
        iq.options_render = {"false: " + f_crit, "true: " + t_crit};
    }
    return iq;
}

ncnn_llm_laya::InternalQ ncnn_llm_laya::parse_question_json(const nlohmann::json& qdef) const {
    LayaQuestion q;
    q.type = qdef["type"].get<std::string>();

    if (qdef["instructions"].is_string()) {
        q.instructions = qdef["instructions"].get<std::string>();
    } else {
        q.instructions = qdef["instructions"].dump();
    }

    if (q.type == "choice") {
        if (qdef.contains("criteria")) {
            auto& crit = qdef["criteria"];
            if (crit.is_array()) {
                for (auto& item : crit) {
                    q.choice_criteria.emplace_back(item.get<std::string>(), "");
                }
            } else if (crit.is_object()) {
                for (auto it = crit.begin(); it != crit.end(); ++it) {
                    std::string desc = it.value().is_null() ? "" : it.value().get<std::string>();
                    q.choice_criteria.emplace_back(it.key(), desc);
                }
            }
        }
    } else if (q.type == "score") {
        if (qdef.contains("criteria") && qdef["criteria"].is_array()) {
            for (auto& item : qdef["criteria"]) {
                q.score_criteria.push_back(item.get<std::string>());
            }
        }
    } else {
        if (qdef.contains("criteria") && qdef["criteria"].is_object()) {
            if (qdef["criteria"].contains("false")) q.noul_false = qdef["criteria"]["false"].get<std::string>();
            if (qdef["criteria"].contains("true")) q.noul_true = qdef["criteria"]["true"].get<std::string>();
        }
    }
    return parse_question(q);
}

void ncnn_llm_laya::build_sequence(const std::string& state,
                                   const InternalQ& q,
                                   std::vector<int>& out_ids,
                                   std::vector<int>& out_markers) const {
    out_ids.clear();
    out_markers.clear();

    int mask_tok_id = tokenizer_->special_ids().mask_id;
    int cls_tok_id = tokenizer_->special_ids().cls_id;
    int sep_tok_id = tokenizer_->special_ids().sep_id;

    std::string mask_tok_str = mask_str_;
    std::string ins_clean = q.ins;
    replace_all(ins_clean, mask_tok_str, " ");

    std::string head_prompt = q.t + " question: " + ins_clean;
    std::vector<int> head_ids = tokenizer_->encode(head_prompt, false, false, false, false);

    std::vector<std::vector<int>> opt_ids;
    opt_ids.reserve(q.options_render.size());
    for (const auto& opt_text : q.options_render) {
        std::string opt_clean = opt_text;
        replace_all(opt_clean, mask_tok_str, " ");
        std::vector<int> t_ids = tokenizer_->encode(" " + opt_clean, false, false, false, false);
        if (t_ids.size() > 48) {
            t_ids.resize(48);
        }
        std::vector<int> cur = {mask_tok_id};
        cur.insert(cur.end(), t_ids.begin(), t_ids.end());
        opt_ids.push_back(std::move(cur));
    }

    size_t total_opts_len = 0;
    for (const auto& o : opt_ids) total_opts_len += o.size();

    int opt_budget = head_max_len_ - static_cast<int>(total_opts_len);
    if (opt_budget < 16 && !opt_ids.empty()) {
        int per = std::max(4, (head_max_len_ - 16) / static_cast<int>(opt_ids.size()));
        total_opts_len = 0;
        for (auto& o : opt_ids) {
            if (static_cast<int>(o.size()) > per) {
                o.resize(per);
            }
            total_opts_len += o.size();
        }
        opt_budget = head_max_len_ - static_cast<int>(total_opts_len);
    }

    int max_head_tokens = std::max(8, opt_budget);
    if (static_cast<int>(head_ids.size()) > max_head_tokens) {
        head_ids.resize(max_head_tokens);
    }

    out_ids.push_back(cls_tok_id);
    out_ids.insert(out_ids.end(), head_ids.begin(), head_ids.end());
    out_ids.push_back(sep_tok_id);

    for (const auto& o : opt_ids) {
        out_markers.push_back(static_cast<int>(out_ids.size()));
        out_ids.insert(out_ids.end(), o.begin(), o.end());
    }
    out_ids.push_back(sep_tok_id);

    int room = std::max(0, max_len_ - static_cast<int>(out_ids.size()) - 1);
    std::string state_clean = state;
    replace_all(state_clean, mask_tok_str, " ");
    std::vector<int> state_ids = tokenizer_->encode(state_clean, false, false, false, false);
    if (static_cast<int>(state_ids.size()) > room) {
        state_ids.resize(room);
    }
    out_ids.insert(out_ids.end(), state_ids.begin(), state_ids.end());
    out_ids.push_back(sep_tok_id);

    if (static_cast<int>(out_ids.size()) > max_len_) {
        out_ids.resize(max_len_);
    }

    // Keep only markers within sequence length
    std::vector<int> valid_markers;
    for (int m : out_markers) {
        if (m < static_cast<int>(out_ids.size())) {
            valid_markers.push_back(m);
        }
    }
    out_markers = std::move(valid_markers);
}


void ncnn_llm_laya::build_sliding_mask(int seq_len, int window, ncnn::Mat& mask_mat) const {
    mask_mat.create(seq_len, seq_len, 1, sizeof(float));
    int half_win = window / 2; // 64
    for (int i = 0; i < seq_len; ++i) {
        float* row = mask_mat.row(i);
        for (int j = 0; j < seq_len; ++j) {
            row[j] = (std::abs(i - j) <= half_win) ? 0.0f : -10000.0f;
        }
    }
}

std::string ncnn_llm_laya::temp_bucket(int qtype, int k) const {
    const char* qtype_names[] = {"choice", "score", "noul"};
    std::string size = (k <= 2) ? "2" : (k <= 5) ? "3-5" : (k <= 10) ? "6-10" : "11+";
    return std::string(qtype_names[qtype]) + ":" + size;
}

float ncnn_llm_laya::confidence_from_probs(const std::vector<float>& p, int k) const {
    if (k < 2) return 1.0f;
    float ent = 0.0f;
    for (int i = 0; i < k; ++i) {
        float prob = std::max(p[i], 1e-12f);
        ent -= prob * std::log(prob);
    }
    return 1.0f - ent / std::log(static_cast<float>(k));
}

LayaAnswer ncnn_llm_laya::evaluate_single_question(const std::string& state,
                                                 const InternalQ& q,
                                                 int& token_count) const {
    LayaAnswer ans;
    ans.type = q.t;

    std::vector<int> seq;
    std::vector<int> markers;
    build_sequence(state, q, seq, markers);

    int L = static_cast<int>(seq.size());
    int K = static_cast<int>(markers.size());
    token_count = L;

    if (K == 0) {
        return ans;
    }

    int qt = (q.t == "choice") ? 0 : (q.t == "score") ? 1 : 2;

    // 1. Build RoPE and Sliding Mask
    ncnn::Mat cos_full, sin_full, cos_slide, sin_slide;
    generate_rope_embed_cache_3d(L, head_dim_, rope_theta_full_, cos_full, sin_full);
    generate_rope_embed_cache_3d(L, head_dim_, rope_theta_slide_, cos_slide, sin_slide);

    ncnn::Mat sliding_mask;
    build_sliding_mask(L, sliding_window_, sliding_mask);

    ncnn::Mat input_ids = ncnn::Mat(L, 1, (void*)seq.data()).clone();

    int qt_val = qt;
    ncnn::Mat qtype_mat = ncnn::Mat(1, (void*)&qt_val).clone();

    // 2. Run Backbone Net
    ncnn::Mat h_mat;
    {
        ncnn::Extractor ex_bb = backbone_net_->create_extractor();
        ex_bb.input("in0", input_ids);
        ex_bb.input("in1", qtype_mat);
        ex_bb.input("in2", sliding_mask);
        ex_bb.input("in3", cos_full);
        ex_bb.input("in4", sin_full);
        ex_bb.input("in5", cos_slide);
        ex_bb.input("in6", sin_slide);

        if (ex_bb.extract("out0", h_mat) != 0) {
            fprintf(stderr, "[ncnn_llm_laya] Backbone extraction failed\n");
            return ans;
        }
    }

    // 3. Extract Marker Features & Run Scorer
    ncnn::Mat m_feats(hidden_size_, K, sizeof(float));
    for (int k = 0; k < K; ++k) {
        int m_idx = markers[k];
        const float* src_row = h_mat.row(m_idx);
        float* dst_row = m_feats.row(k);
        std::memcpy(dst_row, src_row, hidden_size_ * sizeof(float));
    }

    ncnn::Mat logits_mat;
    {
        ncnn::Extractor ex_sc = scorer_net_->create_extractor();
        ex_sc.input("in0", m_feats);
        if (ex_sc.extract("out0", logits_mat) != 0) {
            fprintf(stderr, "[ncnn_llm_laya] Scorer extraction failed\n");
            return ans;
        }
    }

    std::vector<float> logits(K);
    const float* logit_ptr = logits_mat;
    for (int k = 0; k < K; ++k) {
        logits[k] = logit_ptr[k];
    }


    // 4. Calculate Features for Act Head
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_uncal = 0.0f;
    std::vector<float> p_uncal(K);
    for (int k = 0; k < K; ++k) {
        p_uncal[k] = std::exp(logits[k] - max_logit);
        sum_uncal += p_uncal[k];
    }
    for (int k = 0; k < K; ++k) p_uncal[k] /= sum_uncal;

    std::vector<float> p_sorted = p_uncal;
    std::sort(p_sorted.begin(), p_sorted.end(), std::greater<float>());
    float top1 = p_sorted[0];
    float top2 = (K > 1) ? p_sorted[1] : 0.0f;
    float k_float = static_cast<float>(std::max(2, K));
    float ent = 0.0f;
    for (float pu : p_uncal) {
        ent -= pu * std::log(std::max(pu, 1e-9f));
    }
    ent /= std::log(k_float);

    int act_dim = hidden_size_ + 4;
    ncnn::Mat act_in(act_dim, sizeof(float));
    float* act_ptr = act_in;
    const float* pooled_row = h_mat.row(0); // [CLS] / <bos>
    std::memcpy(act_ptr, pooled_row, hidden_size_ * sizeof(float));
    act_ptr[hidden_size_] = top1;
    act_ptr[hidden_size_ + 1] = top1 - top2;
    act_ptr[hidden_size_ + 2] = ent;
    act_ptr[hidden_size_ + 3] = k_float / 255.0f;

    ncnn::Mat act_out_mat;
    {
        ncnn::Extractor ex_act = act_head_net_->create_extractor();
        ex_act.input("in0", act_in);
        if (ex_act.extract("out0", act_out_mat) != 0) {
            fprintf(stderr, "[ncnn_llm_laya] Act head extraction failed\n");
            return ans;
        }
    }

    const float* act_logits = act_out_mat;
    float max_act = std::max(act_logits[0], act_logits[1]);
    float ea0 = std::exp(act_logits[0] - max_act);
    float ea1 = std::exp(act_logits[1] - max_act);
    float act_prob = ea0 / (ea0 + ea1);
    ans.act_probability = std::round(act_prob * 10000.0f) / 10000.0f;

    // 5. Temperature Calibration
    std::string bucket = temp_bucket(qt, K);
    float temp = temperature_[qt];
    auto it_temp = temperature_by_options_.find(bucket);
    if (it_temp != temperature_by_options_.end()) {
        temp = it_temp->second;
    }

    std::vector<float> z(K);
    for (int k = 0; k < K; ++k) z[k] = logits[k] / temp;
    float max_z = *std::max_element(z.begin(), z.end());
    float sum_p = 0.0f;
    std::vector<float> p(K);
    for (int k = 0; k < K; ++k) {
        p[k] = std::exp(z[k] - max_z);
        sum_p += p[k];
    }
    for (int k = 0; k < K; ++k) p[k] /= sum_p;

    ans.confidence = std::round(confidence_from_probs(p, K) * 10000.0f) / 10000.0f;

    if (q.t == "choice") {
        int best_idx = 0;
        float best_prob = p[0];
        for (int k = 1; k < K; ++k) {
            if (p[k] > best_prob) {
                best_prob = p[k];
                best_idx = k;
            }
        }
        ans.choice = q.keys[best_idx];
        for (int k = 0; k < K; ++k) {
            ans.probabilities.emplace_back(q.keys[k], std::round(p[k] * 10000.0f) / 10000.0f);
        }
    } else if (q.t == "score") {
        float expected_score = 0.0f;
        for (int k = 0; k < K; ++k) {
            expected_score += static_cast<float>(k) * p[k];
            ans.probabilities.emplace_back(std::to_string(k), std::round(p[k] * 10000.0f) / 10000.0f);
            ans.legend.emplace_back(std::to_string(k), q.crit_descriptions[k]);
        }
        ans.score = std::round(expected_score * 10000.0f) / 10000.0f;
    } else { // "noul"
        ans.noul = std::round(p[1] * 10000.0f) / 10000.0f;
        ans.probabilities.emplace_back("false", std::round(p[0] * 10000.0f) / 10000.0f);
        ans.probabilities.emplace_back("true", std::round(p[1] * 10000.0f) / 10000.0f);
    }

    return ans;
}

LayaResult ncnn_llm_laya::system_one(const std::string& state,
                                    const std::map<std::string, LayaQuestion>& questions) const {
    LayaResult res;
    res.model = "rl-agent";
    int total_tokens = 0;

    for (const auto& pair : questions) {
        const std::string& qid = pair.first;
        InternalQ iq = parse_question(pair.second);
        int tok_cnt = 0;
        LayaAnswer ans = evaluate_single_question(state, iq, tok_cnt);
        res.answers[qid] = std::move(ans);
        total_tokens += tok_cnt;
    }

    res.input_tokens = total_tokens;
    return res;
}

nlohmann::json ncnn_llm_laya::system_one_json(const std::string& state,
                                             const nlohmann::json& questions_json) const {
    nlohmann::json out;
    out["model"] = "rl-agent";
    nlohmann::json answers_json = nlohmann::json::object();
    int total_tokens = 0;

    for (auto it = questions_json.begin(); it != questions_json.end(); ++it) {
        std::string qid = it.key();
        InternalQ iq = parse_question_json(it.value());
        int tok_cnt = 0;
        LayaAnswer ans = evaluate_single_question(state, iq, tok_cnt);
        total_tokens += tok_cnt;

        nlohmann::json a;
        a["type"] = ans.type;
        if (ans.type == "choice") {
            a["choice"] = ans.choice;
            nlohmann::json probs = nlohmann::json::object();
            for (const auto& p : ans.probabilities) probs[p.first] = p.second;
            a["probabilities"] = probs;
            a["confidence"] = ans.confidence;
            a["rl_agent"] = {{"act_probability", ans.act_probability}};
        } else if (ans.type == "score") {
            a["score"] = ans.score;
            nlohmann::json legend = nlohmann::json::object();
            for (const auto& l : ans.legend) legend[l.first] = l.second;
            a["legend"] = legend;
            nlohmann::json probs = nlohmann::json::object();
            for (const auto& p : ans.probabilities) probs[p.first] = p.second;
            a["probabilities"] = probs;
            a["confidence"] = ans.confidence;
            a["rl_agent"] = {{"act_probability", ans.act_probability}};
        } else {
            a["noul"] = ans.noul;
            a["rl_agent"] = {{"act_probability", ans.act_probability}};
        }
        answers_json[qid] = a;
    }

    out["answers"] = answers_json;
    out["usage"] = {
        {"input_tokens", total_tokens},
        {"output_tokens", 0}
    };
    return out;
}

LayaAnswer ncnn_llm_laya::decide_choice(const std::string& state,
                                      const std::string& instructions,
                                      const std::vector<std::pair<std::string, std::string>>& criteria) const {
    LayaQuestion q;
    q.type = "choice";
    q.instructions = instructions;
    q.choice_criteria = criteria;
    InternalQ iq = parse_question(q);
    int tok = 0;
    return evaluate_single_question(state, iq, tok);
}

LayaAnswer ncnn_llm_laya::decide_choice(const std::string& state,
                                      const std::string& instructions,
                                      const std::vector<std::string>& options) const {
    std::vector<std::pair<std::string, std::string>> crit;
    for (const auto& opt : options) crit.emplace_back(opt, "");
    return decide_choice(state, instructions, crit);
}

LayaAnswer ncnn_llm_laya::decide_score(const std::string& state,
                                     const std::string& instructions,
                                     const std::vector<std::string>& criteria) const {
    LayaQuestion q;
    q.type = "score";
    q.instructions = instructions;
    q.score_criteria = criteria;
    InternalQ iq = parse_question(q);
    int tok = 0;
    return evaluate_single_question(state, iq, tok);
}

LayaAnswer ncnn_llm_laya::decide_noul(const std::string& state,
                                    const std::string& instructions,
                                    const std::string& false_crit,
                                    const std::string& true_crit) const {
    LayaQuestion q;
    q.type = "noul";
    q.instructions = instructions;
    q.noul_false = false_crit;
    q.noul_true = true_crit;
    InternalQ iq = parse_question(q);
    int tok = 0;
    return evaluate_single_question(state, iq, tok);
}

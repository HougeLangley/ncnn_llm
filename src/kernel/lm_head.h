#pragma once

#include <memory>
#include <string>
#include <vector>
#include <mat.h>
#include <net.h>

namespace ncnn {
class Layer;
}

namespace ncnn_llm {

class LlmHead {
public:
    LlmHead();
    ~LlmHead();

    LlmHead(const LlmHead&) = delete;
    LlmHead& operator=(const LlmHead&) = delete;

    // Initialize by sharing weights from an existing embedding Mat (Zero-Copy)
    int init_shared(const std::string& proj_out_param_path,
                    const ncnn::Mat& embed_weight,
                    const ncnn::Option& opt);

    // Initialize from standalone param and bin files (for untied models like minicpm5)
    int init_from_file(const std::string& proj_out_param_path,
                       const std::string& proj_out_bin_path,
                       const ncnn::Option& opt);

    // Run forward pass: hidden_states (1 x hidden_size) -> logits (vocab_size)
    ncnn::Mat forward(const ncnn::Mat& hidden_states, const ncnn::Option& opt) const;

    bool is_shared() const { return is_shared_; }
    bool is_initialized() const { return initialized_; }
    int vocab_size() const { return vocab_size_; }
    int hidden_size() const { return hidden_size_; }

    std::shared_ptr<ncnn::Net> get_net() const { return fallback_net_; }

private:
    bool initialized_ = false;
    bool is_shared_ = false;
    int vocab_size_ = 0;
    int hidden_size_ = 0;

    ncnn::Layer* gemm_layer_ = nullptr;
    ncnn::Mat shared_weight_;
    std::shared_ptr<ncnn::Net> fallback_net_ = nullptr;
};

} // namespace ncnn_llm
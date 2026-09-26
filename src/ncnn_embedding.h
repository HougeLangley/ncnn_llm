#pragma once

#include <string>
#include <vector>
#include <memory>
#include <fstream>

#include <mat.h>
#include <net.h>
#include <nlohmann/json.hpp>

#include "ncnn_llm_base.h"
#include "utils/tokenizer/bpe_tokenizer.h"
#include "utils/tokenizer/unigram_tokenizer.h"
#include "utils/rope_embed.h"

using nlohmann::json;

class ncnn_embedding {
private:
    std::shared_ptr<ncnn::Net> text_encoder_net;
    std::shared_ptr<ncnn::Net> vision_encoder_net;

    std::shared_ptr<BpeTokenizer> bpe_tokenizer;
    std::shared_ptr<UnigramTokenizer> unigram_tokenizer;

    std::string model_type_ = "embedding";
    int text_embed_dim_ = 768;
    int vision_embed_dim_ = 0;
    int rope_head_dim_ = 64;
    float rope_theta_ = 100000.0f;
    std::string rope_type_ = "RoPE";
    int image_size_ = 224;
    std::vector<float> image_mean_ = {0.485f, 0.456f, 0.406f};
    std::vector<float> image_std_ = {0.229f, 0.224f, 0.225f};
    int max_seq_len_ = 512;
    bool use_vulkan_ = false;
    int num_threads_ = 4;
    bool ok_ = true;

public:
    ncnn_embedding(const std::string& model_path, bool use_vulkan = false, int num_threads = 0, int vulkan_device = 0, bool use_bf16 = true);

    std::vector<float> encode_text(const std::string& text) const;
    std::vector<std::vector<float>> encode_text_batch(const std::vector<std::string>& texts) const;
    std::vector<float> encode_image(const std::vector<unsigned char>& image_data, int width, int height, int channels) const;
    std::vector<float> encode_image_file(const std::string& image_path) const;

    bool ok() const { return ok_; }
    int text_embed_dim() const { return text_embed_dim_; }
    int vision_embed_dim() const { return vision_embed_dim_; }
    bool supports_image() const { return vision_encoder_net != nullptr; }

private:
    ncnn::Mat mean_pool(const ncnn::Mat& embeddings) const;
    ncnn::Mat normalize(const ncnn::Mat& vec) const;
    ncnn::Mat preprocess_image(const std::vector<unsigned char>& image_data, int width, int height, int channels) const;
};
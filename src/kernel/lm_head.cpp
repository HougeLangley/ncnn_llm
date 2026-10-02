#if defined(__x86_64__) || defined(_M_X64)
#include "kernel/x86/lm_head_x86.h"
#endif
#include "kernel/lm_head.h"

#include <fstream>
#include <sstream>
#include <iostream>

#include <mat.h>
#include <layer.h>
#include <modelbin.h>
#include <paramdict.h>
#include <cpu.h>
#include "layer_type.h"

namespace ncnn_llm {

LlmHead::LlmHead()
    : gemm_layer_(nullptr), vocab_size_(0), hidden_size_(0),
      initialized_(false), is_shared_(false) {
}

LlmHead::~LlmHead() {
    if (gemm_layer_) {
        delete gemm_layer_;
        gemm_layer_ = nullptr;
    }
}

int LlmHead::init_shared(const std::string& proj_out_param_path,
                         const ncnn::Mat& embed_weight,
                         const ncnn::Option& opt) {
    if (gemm_layer_) {
        delete gemm_layer_;
        gemm_layer_ = nullptr;
    }
    fallback_net_.reset();
    initialized_ = false;
    is_shared_ = false;

    std::ifstream ifs(proj_out_param_path);
    if (!ifs.is_open()) {
        return -1;
    }

    ncnn::ParamDict pd;
    std::string line;
    bool found_gemm = false;

    while (std::getline(ifs, line)) {
        std::istringstream iss(line);
        std::string layer_type;
        if (!(iss >> layer_type)) continue;
        if (layer_type == "Gemm") {
            std::string layer_name;
            int bottom_cnt = 0, top_cnt = 0;
            if (!(iss >> layer_name >> bottom_cnt >> top_cnt)) continue;
            for (int i = 0; i < bottom_cnt + top_cnt; i++) {
                std::string blob_name;
                iss >> blob_name;
            }
            std::string kv;
            while (iss >> kv) {
                auto eq = kv.find('=');
                if (eq != std::string::npos) {
                    int id = std::stoi(kv.substr(0, eq));
                    std::string val_str = kv.substr(eq + 1);
                    if (val_str.find('.') != std::string::npos) {
                        pd.set(id, std::stof(val_str));
                    } else {
                        pd.set(id, std::stoi(val_str));
                    }
                }
            }
            found_gemm = true;
            break;
        }
    }

    if (!found_gemm) {
        return -2;
    }

    vocab_size_ = pd.get(8, 0);
    hidden_size_ = pd.get(9, 0);

    gemm_layer_ = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
    if (!gemm_layer_) {
        return -3;
    }

    gemm_layer_->load_param(pd);

    shared_weight_ = embed_weight;
    ncnn::Mat weights[1] = { shared_weight_ };
    ncnn::ModelBinFromMatArray mb(weights);
    int ret_load = gemm_layer_->load_model(mb);
    if (ret_load != 0) {
        delete gemm_layer_;
        gemm_layer_ = nullptr;
        return ret_load;
    }

#if defined(__riscv)
    // On riscv the bare-layer Gemm only built the fp16s pipeline, but the
    // decoder hidden states arrive as fp32 (extract() deinterleaves), which
    // then falls into the never-built fp32 path and crashes (empty BT_data).
    // Build the pipeline as fp32: LM head is an M=1 bandwidth-bound GEMV,
    // so the fp16 savings do not matter here.
    ncnn::Option opt_fp32 = opt;
    opt_fp32.use_fp16_storage = false;
    opt_fp32.use_fp16_arithmetic = false;
    opt_fp32.use_fp16_packed = false;
    int ret_pipe = gemm_layer_->create_pipeline(opt_fp32);
#else
    int ret_pipe = gemm_layer_->create_pipeline(opt);
#endif
    if (ret_pipe != 0) {
        delete gemm_layer_;
        gemm_layer_ = nullptr;
        return ret_pipe;
    }

    initialized_ = true;
    is_shared_ = true;
    return 0;
}

int LlmHead::init_from_file(const std::string& proj_out_param_path,
                            const std::string& proj_out_bin_path,
                            const ncnn::Option& opt) {
    if (gemm_layer_) {
        delete gemm_layer_;
        gemm_layer_ = nullptr;
    }
    fallback_net_.reset();
    initialized_ = false;
    is_shared_ = false;

    fallback_net_ = std::make_shared<ncnn::Net>();
    fallback_net_->opt = opt;
    int ret = fallback_net_->load_param(proj_out_param_path.c_str());
    if (ret != 0) return ret;
    ret = fallback_net_->load_model(proj_out_bin_path.c_str());
    if (ret != 0) return ret;

    initialized_ = true;
    is_shared_ = false;
    return 0;
}

ncnn::Mat LlmHead::forward(const ncnn::Mat& hidden_states, const ncnn::Option& opt) const {
    if (gemm_layer_) {
#if defined(__x86_64__) || defined(_M_X64)
        const int M = hidden_states.dims == 1 ? 1 : hidden_states.h;
        const int K = hidden_states.w;
        if (M == 1 && is_shared_ && !shared_weight_.empty() && K == hidden_size_) {
            const int threads = opt.num_threads > 0 ? opt.num_threads : ncnn::get_cpu_count();
            if (shared_weight_.elemsize == sizeof(unsigned short)) {
                if (hidden_states.elemsize == sizeof(unsigned short)) {
                    ncnn::Mat logits(vocab_size_, 1, 4u, opt.blob_allocator);
                    gemv_bf16_x86((const unsigned short*)hidden_states,
                                  (const unsigned short*)shared_weight_,
                                  (float*)logits, vocab_size_, hidden_size_, threads);
                    return logits;
                } else if (hidden_states.elemsize == sizeof(float)) {
                    ncnn::Mat logits(vocab_size_, 1, 4u, opt.blob_allocator);
                    gemv_bf16_fp32_x86((const float*)hidden_states,
                                       (const unsigned short*)shared_weight_,
                                       (float*)logits, vocab_size_, hidden_size_, threads);
                    return logits;
                }
            } else if (shared_weight_.elemsize == sizeof(float) && hidden_states.elemsize == sizeof(float)) {
                ncnn::Mat logits(vocab_size_, 1, 4u, opt.blob_allocator);
                gemv_fp32_x86((const float*)hidden_states, (const float*)shared_weight_, (float*)logits,
                              vocab_size_, hidden_size_, threads);
                return logits;
            }
        }
#endif
        ncnn::Mat in_blob = hidden_states;
        if (in_blob.dims == 1) {
            in_blob = in_blob.reshape(in_blob.w, 1);
        }
        std::vector<ncnn::Mat> bottom_blobs = { in_blob };
        std::vector<ncnn::Mat> top_blobs(1);
        gemm_layer_->forward(bottom_blobs, top_blobs, opt);
        return top_blobs[0];
    }
    if (fallback_net_) {
        ncnn::Extractor ex = fallback_net_->create_extractor();
        ex.input("in0", hidden_states);
        ncnn::Mat logits;
        ex.extract("out0", logits);
        return logits;
    }
    return ncnn::Mat();
}

} // namespace ncnn_llm
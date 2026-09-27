// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GDR_H
#define LAYER_GDR_H

#include <vector>
#include <mat.h>
#include <layer.h>
#include <net.h>
#include "kernel/shortconv.h"

namespace ncnn {

class GatedDeltaRule : public Layer
{
public:
    GatedDeltaRule();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

public:
    int num_k_heads;
    int num_v_heads;
};

} // namespace ncnn

using ncnn::GatedDeltaRule;

void register_gdr_layers(ncnn::Net& net, bool force_naive = false);

#endif // LAYER_GDR_H

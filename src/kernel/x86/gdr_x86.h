// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GDR_X86_H
#define LAYER_GDR_X86_H

#include "kernel/gdr.h"

namespace ncnn {

class GatedDeltaRule_x86 : public GatedDeltaRule
{
public:
    GatedDeltaRule_x86();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

using ncnn::GatedDeltaRule_x86;

#endif // LAYER_GDR_X86_H

// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SHORTCONV_X86_H
#define LAYER_SHORTCONV_X86_H

#include "kernel/shortconv.h"

namespace ncnn {

class ShortConv_x86 : public ShortConv
{
public:
    ShortConv_x86();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

using ncnn::ShortConv_x86;

#endif // LAYER_SHORTCONV_X86_H

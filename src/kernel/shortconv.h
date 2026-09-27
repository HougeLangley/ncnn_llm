// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SHORTCONV_H
#define LAYER_SHORTCONV_H

#include <vector>
#include <mat.h>
#include <layer.h>

namespace ncnn {

class ShortConv : public Layer
{
public:
    ShortConv();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
};

} // namespace ncnn

using ncnn::ShortConv;

#endif // LAYER_SHORTCONV_H

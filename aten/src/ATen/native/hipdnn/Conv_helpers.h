#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <array>
#include <optional>
#include <tuple>

// hipDNN convolution implementation helpers. Conv_dispatch.cpp's at:: op
// definitions and backward stub registrations call into these when
// torch.backends.miopen.use_hipdnn is enabled and the build includes hipDNN
// (USE_HIPDNN). On builds without hipDNN, these are not declared and callers
// guard with #ifdef USE_HIPDNN.

namespace at::native {

at::Tensor hipdnn_convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic);

at::Tensor hipdnn_convolution_transpose(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic);

std::tuple<at::Tensor, at::Tensor, at::Tensor> hipdnn_convolution_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    std::array<bool, 3> output_mask);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
hipdnn_convolution_transpose_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    std::array<bool, 3> output_mask);

} // namespace at::native

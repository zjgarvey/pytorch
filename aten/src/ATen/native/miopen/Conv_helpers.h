#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <array>
#include <optional>
#include <tuple>

// MIOpen convolution implementation helpers. Conv_dispatch.cpp's at:: op
// definitions and backward stub registrations call into these. They live in
// Conv_miopen.cpp; on builds without MIOpen they're stubs that throw.

namespace at::native {

at::Tensor miopen_convolution_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic);

at::Tensor miopen_convolution_transpose_impl(
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

at::Tensor miopen_depthwise_convolution_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic);

std::tuple<at::Tensor, at::Tensor, at::Tensor> miopen_convolution_backward_impl(
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
miopen_convolution_transpose_backward_impl(
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

std::tuple<at::Tensor, at::Tensor, at::Tensor>
miopen_depthwise_convolution_backward_impl(
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

} // namespace at::native

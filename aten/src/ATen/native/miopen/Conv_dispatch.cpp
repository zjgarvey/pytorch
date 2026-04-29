// Conv_dispatch.cpp routes the at:: miopen_convolution* ops and the matching
// backward stubs between MIOpen (Conv_miopen.cpp's *_impl helpers) and hipDNN
// (Conv_hipdnn.cpp's hipdnn_* helpers). The runtime toggle is
// at::globalContext().userMiopenUseHipdnn() (exposed to Python as
// torch.backends.miopen.use_hipdnn). On builds without USE_HIPDNN the hipDNN
// branch is compiled out and every call goes to the MIOpen impl.
//
// TODO(post-merge): replace shouldUseHipdnn's static filter with a runtime
// graph-capability probe via detail::getCUDAHooks().canHipDNNHandleConv(...).
// See agent_space/hipdnn_post_refactor_todos.md.

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/miopen/Conv_helpers.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/miopen_convolution_native.h>
#include <ATen/ops/miopen_convolution_transpose_native.h>
#include <ATen/ops/miopen_depthwise_convolution_native.h>
#endif

#ifdef USE_HIPDNN
#include <ATen/native/hipdnn/Conv_helpers.h>
#endif

namespace at::native {

namespace {

// Static filter: would hipDNN even be willing to try this shape/dtype?
// Mirrors the gating in the original hipdnn-peer-backend predicate. Will be
// replaced by a runtime probe in a follow-up patch (see file-level TODO).
[[maybe_unused]] bool shouldUseHipdnn(
    const Tensor& input,
    const Tensor& /*weight*/,
    bool deterministic) {
  if (deterministic) {
    // hipDNN does not yet provide engine-level determinism guarantees; fall
    // through to MIOpen rather than refusing the call.
    return false;
  }
  auto dtype = input.scalar_type();
  if (dtype != at::kFloat && dtype != at::kHalf && dtype != at::kBFloat16) {
    return false;
  }
  if (input.dim() < 4 || input.dim() > 5) {
    return false;
  }
  return true;
}

} // namespace

Tensor miopen_convolution(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
#ifdef USE_HIPDNN
  if (at::globalContext().userMiopenUseHipdnn() &&
      shouldUseHipdnn(input, weight, deterministic)) {
    return hipdnn_convolution(
        input, weight, bias_opt, padding, stride, dilation,
        groups, benchmark, deterministic);
  }
#endif
  return miopen_convolution_impl(
      input, weight, bias_opt, padding, stride, dilation,
      groups, benchmark, deterministic);
}

Tensor miopen_convolution_transpose(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
#ifdef USE_HIPDNN
  if (at::globalContext().userMiopenUseHipdnn() &&
      shouldUseHipdnn(input, weight, deterministic)) {
    return hipdnn_convolution_transpose(
        input, weight, bias_opt, padding, output_padding, stride, dilation,
        groups, benchmark, deterministic);
  }
#endif
  return miopen_convolution_transpose_impl(
      input, weight, bias_opt, padding, output_padding, stride, dilation,
      groups, benchmark, deterministic);
}

Tensor miopen_depthwise_convolution(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
#ifdef USE_HIPDNN
  // hipDNN has no separate depthwise path; regular hipdnn_convolution handles
  // the depthwise case via group counts inferred from tensor shapes.
  if (at::globalContext().userMiopenUseHipdnn() &&
      shouldUseHipdnn(input, weight, deterministic)) {
    return hipdnn_convolution(
        input, weight, bias_opt, padding, stride, dilation,
        groups, benchmark, deterministic);
  }
#endif
  return miopen_depthwise_convolution_impl(
      input, weight, bias_opt, padding, stride, dilation,
      groups, benchmark, deterministic);
}

namespace {

std::tuple<Tensor, Tensor, Tensor> miopen_convolution_backward_router(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    std::array<bool, 3> output_mask) {
#ifdef USE_HIPDNN
  if (at::globalContext().userMiopenUseHipdnn() &&
      shouldUseHipdnn(input, weight, deterministic)) {
    return hipdnn_convolution_backward(
        input, grad_output, weight, padding, stride, dilation,
        groups, benchmark, deterministic, output_mask);
  }
#endif
  return miopen_convolution_backward_impl(
      input, grad_output, weight, padding, stride, dilation,
      groups, benchmark, deterministic, output_mask);
}

std::tuple<Tensor, Tensor, Tensor> miopen_convolution_transpose_backward_router(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    std::array<bool, 3> output_mask) {
#ifdef USE_HIPDNN
  if (at::globalContext().userMiopenUseHipdnn() &&
      shouldUseHipdnn(input, weight, deterministic)) {
    return hipdnn_convolution_transpose_backward(
        input, grad_output, weight, padding, output_padding, stride, dilation,
        groups, benchmark, deterministic, output_mask);
  }
#endif
  return miopen_convolution_transpose_backward_impl(
      input, grad_output, weight, padding, output_padding, stride, dilation,
      groups, benchmark, deterministic, output_mask);
}

std::tuple<Tensor, Tensor, Tensor> miopen_depthwise_convolution_backward_router(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    std::array<bool, 3> output_mask) {
#ifdef USE_HIPDNN
  if (at::globalContext().userMiopenUseHipdnn() &&
      shouldUseHipdnn(input, weight, deterministic)) {
    return hipdnn_convolution_backward(
        input, grad_output, weight, padding, stride, dilation,
        groups, benchmark, deterministic, output_mask);
  }
#endif
  return miopen_depthwise_convolution_backward_impl(
      input, grad_output, weight, padding, stride, dilation,
      groups, benchmark, deterministic, output_mask);
}

} // namespace

REGISTER_CUDA_DISPATCH(
    miopen_convolution_backward_stub,
    &miopen_convolution_backward_router)
REGISTER_CUDA_DISPATCH(
    miopen_convolution_transpose_backward_stub,
    &miopen_convolution_transpose_backward_router)
REGISTER_CUDA_DISPATCH(
    miopen_depthwise_convolution_backward_stub,
    &miopen_depthwise_convolution_backward_router)

} // namespace at::native

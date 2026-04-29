#pragma once

#include <ATen/Tensor.h>
#include <ATen/miopen/miopen-wrapper.h>
#include <c10/macros/Export.h>

namespace at::native {

// Generic dtype mapping for ROCm DNN libraries (MIOpen, hipDNN, ...).
// Each library specializes getDataType<LibDtype> for its enum type in the
// translation unit that owns the library's includes.
template <typename LibDtype>
LibDtype getDataType(const at::Tensor& tensor);

TORCH_CUDA_CPP_API miopenDataType_t getMiopenDataType(const at::Tensor& tensor);

int64_t miopen_version();

} // namespace at::native

#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>
#include <cudnn_frontend.h>

#include <memory>

namespace at::native {

inline cudnn_frontend::DataType_t getCudnnFEDataType(const at::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    case at::kFloat:
      return cudnn_frontend::DataType_t::FLOAT;
    case at::kHalf:
      return cudnn_frontend::DataType_t::HALF;
    case at::kBFloat16:
      return cudnn_frontend::DataType_t::BFLOAT16;
    default:
      TORCH_CHECK(
          false,
          "getCudnnFEDataType() not supported for ",
          toString(tensor.scalar_type()));
  }
}

inline std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
createTensorAttributes(const Tensor& t) {
  auto tensor = std::make_shared<cudnn_frontend::graph::Tensor_attributes>();
  tensor->set_dim(t.sizes().vec()).set_data_type(getCudnnFEDataType(t));
  tensor->set_stride(t.strides().vec());
  return tensor;
}

} // namespace at::native

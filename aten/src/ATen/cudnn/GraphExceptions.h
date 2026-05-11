#pragma once

#include <c10/util/Exception.h>
#include <cudnn_frontend.h>

namespace c10 {

class CudnnFEError : public c10::Error {
  using Error::Error;
};

} // namespace c10

// Asserts that an expression returning a cudnn_frontend::error_t is_good();
// raises a CudnnFEError with the frontend's error message otherwise.
#define CUDNN_FE_CHECK(EXPR)             \
  do {                                   \
    auto error_object = EXPR;            \
    if (!error_object.is_good()) {       \
      TORCH_CHECK_WITH(                  \
          CudnnFEError,                  \
          false,                         \
          "cuDNN Frontend error: ",      \
          error_object.get_message());   \
    }                                    \
  } while (0)

#include <ATen/hip/detail/DeviceThreadHandles.h>
#include <ATen/miopen/Handle.h>
#include <c10/hip/HIPStream.h>

#include <ATen/hip/Exceptions.h>
#include <ATen/miopen/Exceptions.h>

namespace at::native {

template <>
void HandleTraits<miopenHandle_t>::create(miopenHandle_t* handle) {
  MIOPEN_CHECK(miopenCreate(handle));
}

template <>
void HandleTraits<miopenHandle_t>::destroy(miopenHandle_t /*handle*/) {
  // this is because of something dumb in the ordering of
  // destruction. Sometimes atexit, the cuda context (or something)
  // would already be destroyed by the time this gets destroyed. It
  // happens in fbcode setting. @colesbury and I decided to not destroy
  // the handle as a workaround.
  //   - @soumith
  //
  // Further note: this is now disabled globally, because we are seeing
  // the same issue as mentioned above in CUDA 11 CI.
  //   - @zasdfgbnm
}

template <>
void HandleTraits<miopenHandle_t>::setCurrentStream(miopenHandle_t handle) {
  MIOPEN_CHECK(miopenSetStream(handle, at::cuda::getCurrentCUDAStream()));
}

miopenHandle_t getMiopenHandle() {
  return getHandle<miopenHandle_t>();
}

} // namespace at::native

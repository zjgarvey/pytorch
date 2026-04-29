#pragma once

#include <ATen/hip/Exceptions.h>
#include <ATen/hip/detail/DeviceThreadHandles.h>
#include <ATen/miopen/miopen-wrapper.h>
#include <c10/core/Device.h>
#include <c10/hip/HIPFunctions.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#include <memory>

namespace at::native {

// Generic handle traits for ROCm DNN libraries (MIOpen, hipDNN, ...).
// Each library specializes HandleTraits<H> with create / destroy /
// setCurrentStream in its own translation unit (Handle.cpp for MIOpen,
// Conv_hipdnn.cpp for hipDNN). setCurrentStream binds the current stream
// itself, keeping the templated getHandle<H>() free of per-library stream
// type churn.
template <typename Handle>
struct HandleTraits {
  static void create(Handle*);
  static void destroy(Handle);
  static void setCurrentStream(Handle);
};

// Returns a handle for the current device, with the current stream bound.
// The static pool is per-(translation-unit, type) due to
// DeviceThreadHandlePool's anonymous-namespace declaration; non-templated
// library entry points (getMiopenHandle, etc.) wrap this function in
// their respective .cpp so each library has exactly one pool program-wide.
template <typename Handle>
Handle getHandle() {
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

  using Pool = at::cuda::DeviceThreadHandlePool<
      Handle,
      &HandleTraits<Handle>::create,
      &HandleTraits<Handle>::destroy>;
  // Thread local PoolWindows are lazily-initialized to avoid initialization
  // issues that caused hangs on Windows. See pytorch/pytorch#22405. The
  // unique_ptrs are destroyed when the thread terminates, releasing reserved
  // handles back to the pool.
  static auto pool = std::make_shared<Pool>();
  thread_local std::unique_ptr<typename Pool::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  HandleTraits<Handle>::setCurrentStream(handle);
  return handle;
}

TORCH_CUDA_CPP_API miopenHandle_t getMiopenHandle();

} // namespace at::native

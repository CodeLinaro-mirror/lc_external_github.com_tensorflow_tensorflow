/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDART_KERNEL_REGISTRY_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDART_KERNEL_REGISTRY_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/base/casts.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/kernel_spec.h"

namespace stream_executor::cuda {

struct CudaRuntimeKernel {
  // The CUBIN (ELF) for a single GPU architecture, extracted from the captured
  // fatbin. This points into memory owned by the process-wide registry and
  // stays valid for the lifetime of the process.
  absl::Span<const uint8_t> cubin;
  // The mangled device symbol name of the kernel.
  absl::string_view name;
};

// Statically-extracted CUDA kernel function attributes.
//
// These values are read directly from the compiled CUBIN (ELF) for a specific
// target compute capability, without a GPU, a CUDA context, or loading the
// module. They contain only the information that can be determined statically
// and are intended as inputs to CUDA occupancy calculations.
struct CudaKernelFuncAttributes {
  // The compute capability whose CUBIN these values were extracted from.
  CudaComputeCapability compute_capability;
  // Number of registers used by each thread.
  int num_regs = 0;
  // Statically-allocated shared memory in bytes.
  size_t static_shared_size_bytes = 0;
  // User `__constant__` memory associated with the kernel in bytes.
  // Best-effort: derived from `.nv.constant*` sections.
  size_t const_size_bytes = 0;
  // Stack frame / local memory per thread in bytes.
  // Best-effort: derived from the `.nv.info` frame-size attribute.
  size_t local_size_bytes = 0;
  // Number of named barriers used by the kernel.
  // Best-effort: derived from the `.text` section flags.
  int num_barriers = 0;
  // The maximum threads per block requested via `__launch_bounds__`, if any.
  // `std::nullopt` if the kernel does not declare launch bounds.
  std::optional<int> max_threads_per_block;
};

// Returns the single-architecture CUBIN and mangled kernel symbol name of the
// kernel that has been registered in the CUDA runtime with the given host
// function pointer, selecting the CUBIN for `compute_capability`.
//
// Only the CUBIN for the requested architecture is returned (a sub-span of the
// captured fatbin), rather than the whole fatbin.
//
// Returns an error status if the kernel is not found, or if the captured fatbin
// does not contain a CUBIN for the requested architecture (for example, because
// it is PTX-only or was compressed; see embeddable_cuda_library).
//
// Usage:
// Assuming you would call a CUDA kernel like this:
//   MyKernel<<<1, 2, 3>>>(a, b, c);
// where `MyKernel` is a CUDA C++ kernel that has been linked into the binary,
// then you can get a KernelLoaderSpec for the kernel by calling:
//   auto cubin_spec = FindCudaRuntimeKernel(MyKernel, cc);
absl::StatusOr<CudaRuntimeKernel> FindCudaRuntimeKernel(
    const void* host_fun, const CudaComputeCapability& compute_capability);

// Returns statically-extracted function attributes for the kernel registered
// with the given host function pointer, read from the CUBIN for
// `compute_capability`.
//
// Returns an error status if the kernel or its CUBIN for the requested
// architecture cannot be found or parsed.
absl::StatusOr<CudaKernelFuncAttributes> FindCudaRuntimeKernelFuncAttributes(
    const void* host_fun, const CudaComputeCapability& compute_capability);

template <typename ReturnT, typename... Args>
absl::StatusOr<KernelLoaderSpec> FindCudaRuntimeKernel(
    ReturnT (*host_fun)(Args...),
    const CudaComputeCapability& compute_capability) {
  ABSL_ASSIGN_OR_RETURN(CudaRuntimeKernel kernel,
                   FindCudaRuntimeKernel(absl::bit_cast<const void*>(host_fun),
                                         compute_capability));
  return KernelLoaderSpec::CreateCudaCubinInMemorySpec(
      kernel.cubin, std::string(kernel.name), sizeof...(Args));
}

template <typename ReturnT, typename... Args>
absl::StatusOr<CudaKernelFuncAttributes> FindCudaRuntimeKernelFuncAttributes(
    ReturnT (*host_fun)(Args...),
    const CudaComputeCapability& compute_capability) {
  return FindCudaRuntimeKernelFuncAttributes(
      absl::bit_cast<const void*>(host_fun), compute_capability);
}

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDART_KERNEL_REGISTRY_H_

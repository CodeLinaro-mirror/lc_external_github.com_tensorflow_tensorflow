/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_BACKENDS_GPU_RUNTIME_BUFFER_DEBUG_LOG_STRUCTS_H_
#define XLA_BACKENDS_GPU_RUNTIME_BUFFER_DEBUG_LOG_STRUCTS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "Eigen/Core"
#include "xla/backends/gpu/runtime/buffer_debug_log.pb.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla::gpu {

TSL_LIB_GTL_DEFINE_INT_TYPE(BufferDebugLogEntryId, uint32_t)

struct BufferDebugLogEntry {
  // An ID that uniquely identifies a log entry within a HLO module execution.
  BufferDebugLogEntryId entry_id;
  uint32_t value;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const BufferDebugLogEntry& entry) {
    absl::Format(&sink, "{entry_id: %v, value: %u}", entry.entry_id.value(),
                 entry.value);
  }

  bool operator==(const BufferDebugLogEntry& other) const {
    return std::tie(entry_id, value) == std::tie(other.entry_id, other.value);
  }

  bool operator!=(const BufferDebugLogEntry& other) const {
    return !(*this == other);
  }
};

// The struct layout must match on both host and device.
static_assert(_Alignof(BufferDebugLogEntry) == _Alignof(uint32_t));
static_assert(sizeof(BufferDebugLogEntry) == sizeof(uint32_t) * 2);
static_assert(offsetof(BufferDebugLogEntry, entry_id) == 0);
static_assert(offsetof(BufferDebugLogEntry, value) == sizeof(uint32_t));

// All kinds of floats supported by the float checker. Determines which fields
// of FloatValue unions are initialized within FloatCheckResult.
enum FloatType : uint32_t { kFloat, kBFloat16 };

template <typename Sink>
void AbslStringify(Sink& sink, FloatType type) {
  switch (type) {
    case kFloat:
      absl::Format(&sink, "F32");
      break;
    case kBFloat16:
      absl::Format(&sink, "BF16");
      break;
  }
}

// A union of all supported float types. Used to enable reusing the same
// BufferDebugLog for buffers with different float types.
union FloatValue {
  // Default constructor is necessary to make FloatCheckResult usable in
  // std::vector. However, using it from a CUDA kernel code requires __host__
  // __device__ attributes, otherwise the build fails.
#if defined(__CUDACC__) || defined(__HIPCC__)
  __host__ __device__
#endif
  FloatValue()
      : f32(0.0f) {
  }

  float f32;
  Eigen::bfloat16 bf16;

  // Caller must ensure `type` matches the initialized field of the union.
  std::string ToStringAs(FloatType type) const {
    switch (type) {
      case kFloat:
        return absl::StrFormat("%f (F32)", f32);
      case kBFloat16:
        return absl::StrFormat("%f (BF16)", static_cast<float>(bf16));
    }
    return "Unsupported type";
  }
};

static_assert(_Alignof(FloatValue) == _Alignof(float));
static_assert(sizeof(FloatValue) == sizeof(float));
static_assert(offsetof(FloatValue, f32) == 0);
static_assert(offsetof(FloatValue, bf16) == 0);

// Output of float checker for a single buffer.
struct FloatCheckResult {
  uint32_t nan_count;
  uint32_t inf_count;
  uint32_t zero_count;
  FloatType float_type;  // Determines which fields of min/max_value are valid.
  FloatValue min_value;
  FloatValue max_value;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const FloatCheckResult& result) {
    absl::Format(
        &sink,
        "{nan_count: %u, inf_count: %u, zero_count: %u, min_value: %s, "
        "max_value: %s}",
        result.nan_count, result.inf_count, result.zero_count,
        result.min_value.ToStringAs(result.float_type),
        result.max_value.ToStringAs(result.float_type));
  }

  template <typename T>
  absl::StatusOr<T> GetMinValue() const {
    return GetValueIfTypeMatches<T>(min_value);
  }

  template <typename T>
  absl::StatusOr<T> GetMaxValue() const {
    return GetValueIfTypeMatches<T>(max_value);
  }

 private:
  template <typename T>
  absl::StatusOr<T> GetValueIfTypeMatches(const FloatValue& value) const {
    switch (float_type) {
      case kFloat:
        if constexpr (std::is_same_v<T, float>) {
          return value.f32;
        } else {
          return absl::InvalidArgumentError(
              absl::StrFormat("Attempted to read %v as F32", float_type));
        }
      case kBFloat16:
        if constexpr (std::is_same_v<T, Eigen::bfloat16>) {
          return value.bf16;
        } else {
          return absl::InvalidArgumentError(
              absl::StrFormat("Attempted to read %v as BF16", float_type));
        }
    }
    return absl::InternalError(
        absl::StrFormat("Unsupported float type: %v", float_type));
  }
};

// The struct layout must match on both host and device.
static_assert(_Alignof(FloatCheckResult) == _Alignof(uint32_t));
static_assert(sizeof(FloatCheckResult) ==
              sizeof(uint32_t) * 4 + sizeof(FloatValue) * 2);
static_assert(offsetof(FloatCheckResult, nan_count) == 0);
static_assert(offsetof(FloatCheckResult, inf_count) == sizeof(uint32_t));
static_assert(offsetof(FloatCheckResult, zero_count) == sizeof(uint32_t) * 2);
static_assert(offsetof(FloatCheckResult, float_type) == sizeof(uint32_t) * 3);
static_assert(offsetof(FloatCheckResult, min_value) == sizeof(uint32_t) * 4);
static_assert(offsetof(FloatCheckResult, max_value) ==
              sizeof(uint32_t) * 4 + sizeof(FloatValue));

// A single entry in the BufferDebugLog for float checks.
struct BufferDebugFloatCheckEntry {
  // An ID that uniquely identifies a log entry within a HLO module execution.
  BufferDebugLogEntryId entry_id;
  FloatCheckResult result;

  template <typename Sink>
  friend void AbslStringify(Sink& sink,
                            const BufferDebugFloatCheckEntry& entry) {
    absl::Format(&sink, "{entry_id: %v, result: %v}", entry.entry_id.value(),
                 entry.result);
  }
};

// The struct layout must match on both host and device.
static_assert(_Alignof(BufferDebugFloatCheckEntry) == _Alignof(uint32_t));
static_assert(sizeof(BufferDebugFloatCheckEntry) ==
              sizeof(uint32_t) + sizeof(FloatCheckResult));
static_assert(offsetof(BufferDebugFloatCheckEntry, entry_id) == 0);
static_assert(offsetof(BufferDebugFloatCheckEntry, result) == sizeof(uint32_t));

struct BufferDebugLogHeader {
  // The first entry in `BufferDebugLogEntry` following the header that has not
  // been written to. May be bigger than `capacity` if the log was truncated.
  uint32_t write_idx;
  // The number of `BufferDebugLogEntry` structs the log can hold.
  uint32_t capacity;
};

// The struct layout must match on both host and device.
static_assert(_Alignof(BufferDebugLogHeader) == _Alignof(uint32_t));
static_assert(sizeof(BufferDebugLogHeader) == sizeof(uint32_t) * 2);
static_assert(offsetof(BufferDebugLogHeader, write_idx) == 0);
static_assert(offsetof(BufferDebugLogHeader, capacity) == sizeof(uint32_t));

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_BUFFER_DEBUG_LOG_STRUCTS_H_

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

#ifndef XLA_BACKENDS_AUTOTUNER_FILE_BASED_AUTOTUNER_CACHE_H_
#define XLA_BACKENDS_AUTOTUNER_FILE_BASED_AUTOTUNER_CACHE_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/autotuner/autotune_config.h"
#include "xla/backends/autotuner/autotuner_cache.pb.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/stream_executor/device_description.h"

namespace xla {

// File-based implementation of the AutotunerCacheInterface.
// This class stores autotuner cache entries in the file system. Each cache
// entry is stored as a separate textproto file in a directory specified by the
// AutotuneConfig. The cache directory can be on any file system supported by
// the TSL environment (tsl::Env).
//
// The filename is derived from a SHA256 hash of the cache key, which
// incorporates the HLO instruction, device description, and cache version.
// This approach avoids excessively large cache files and reduces the likelihood
// of corruption. A temporary directory within the cache directory is used
// during write operations to prevent partially written files from being read.
class FileBasedAutotunerCache : public AutotunerCacheInterface {
 public:
  static absl::StatusOr<std::unique_ptr<AutotunerCacheInterface>> Create(
      const AutotuneConfig& autotune_config,
      const se::DeviceDescription& device_desc, int version);

  std::optional<AutotunerCacheEntry> Lookup(
      const HloInstruction* instr) override ABSL_LOCKS_EXCLUDED(mutex_);

  absl::Status Insert(const HloInstruction* instr,
                      AutotunerCacheEntry& entry) override
      ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  FileBasedAutotunerCache(const AutotuneConfig& autotune_config,
                          const se::DeviceDescription& device_desc,
                          int version);

  static std::string DeviceDescriptionToString(
      const se::DeviceDescription& device_desc);

  absl::StatusOr<std::string> GetMapKey(const HloInstruction* instr);

  absl::StatusOr<AutotunerCacheKey> GetProtoKey(const HloInstruction* instr);

  absl::StatusOr<std::string> GetCacheFilePath(absl::string_view map_key);

  std::string GetCacheFilePattern();

  absl::Status Load() ABSL_LOCKS_EXCLUDED(mutex_);

  absl::Status Save(absl::string_view map_key, const AutotunerCacheEntry& entry)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  AutotuneConfig autotune_config_;
  const se::DeviceDescription device_desc_;
  const int version_;
  absl::Mutex mutex_;
  absl::flat_hash_map<std::string, AutotunerCacheEntry> in_memory_cache_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_FILE_BASED_AUTOTUNER_CACHE_H_

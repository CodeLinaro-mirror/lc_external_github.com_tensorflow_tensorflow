/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/autotuner/tiered_cache.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/local_cache.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

TieredCache::TieredCache(
    AutotuneScope scope,
    std::unique_ptr<AutotunerCacheInterface> persistent_cache)
    : local_cache_(
          std::make_unique<LocalCache>(persistent_cache->GetKeyMatchingMode(),
                                       &LocalCacheStorage::GetInstance(scope))),
      persistent_cache_(std::move(persistent_cache)) {}

std::optional<AutotunerCacheInterface::Config> TieredCache::Lookup(
    const HloInstruction* instr) {
  // Query the local in-memory cache first.
  std::optional<Config> config = local_cache_->Lookup(instr);
  if (config.has_value()) {
    return config;
  }

  // On a local cache miss, query the persistent cache.
  config = persistent_cache_->Lookup(instr);
  if (config.has_value()) {
    // Populate the local in-memory cache for subsequent queries.
    local_cache_->Insert(instr, *config).IgnoreError();
    return config;
  }

  return std::nullopt;
}

absl::Status TieredCache::Insert(const HloInstruction* instr,
                                 const Config& config) {
  // Write to both the local and persistent caches.
  absl::Status local_status = local_cache_->Insert(instr, config);
  absl::Status persistent_status = persistent_cache_->Insert(instr, config);

  if (!local_status.ok()) {
    return local_status;
  }
  return persistent_status;
}

AutotunerCacheInterface::CacheStats TieredCache::GetCacheStats() const {
  CacheStats local_stats = local_cache_->GetCacheStats();
  CacheStats persistent_stats = persistent_cache_->GetCacheStats();
  CacheStats stats;
  stats.hits = local_stats.hits + persistent_stats.hits;
  stats.misses = persistent_stats.misses;
  return stats;
}

absl::StatusOr<std::string> TieredCache::Serialize(
    absl::Span<const HloInstruction* const> instructions_to_serialize) {
  return persistent_cache_->Serialize(instructions_to_serialize);
}

absl::Status TieredCache::Deserialize(absl::string_view serialized_cache) {
  // Update the persistent cache.
  RETURN_IF_ERROR(persistent_cache_->Deserialize(serialized_cache));
  // Invalidate or update local cache by deserializing into it as well.
  return local_cache_->Deserialize(serialized_cache);
}

}  // namespace xla

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

#ifndef XLA_BACKENDS_AUTOTUNER_TIERED_CACHE_H_
#define XLA_BACKENDS_AUTOTUNER_TIERED_CACHE_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// TieredCache combines adds a local in-memory cache and a persistent cache
// into a single interface, to avoid repeated queries to the persistent cache.
//
// Read requests check the local cache first. On a miss, they query the
// persistent cache and populate the local cache with the result on a hit.
// Write requests update both the local and persistent caches.
class TieredCache : public AutotunerCacheInterface {
 public:
  TieredCache(AutotuneScope scope,
              std::unique_ptr<AutotunerCacheInterface> persistent_cache);

  ~TieredCache() override = default;

  std::optional<Config> Lookup(const HloInstruction* instr) override;

  absl::Status Insert(const HloInstruction* instr,
                      const Config& config) override;

  CacheStats GetCacheStats() const override;

  absl::StatusOr<std::string> Serialize(absl::Span<const HloInstruction* const>
                                            instructions_to_serialize) override;

  absl::Status Deserialize(absl::string_view serialized_cache) override;

  CacheMode GetMode() const override { return persistent_cache_->GetMode(); }
  KeyMatchingMode GetKeyMatchingMode() const override {
    return persistent_cache_->GetKeyMatchingMode();
  }

 private:
  std::unique_ptr<AutotunerCacheInterface> local_cache_;
  std::unique_ptr<AutotunerCacheInterface> persistent_cache_;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_TIERED_CACHE_H_

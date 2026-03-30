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

#ifndef XLA_HLO_TRANSFORMS_COLLECTIVES_ASYNC_COLLECTIVE_DESTROYER_H_
#define XLA_HLO_TRANSFORMS_COLLECTIVES_ASYNC_COLLECTIVE_DESTROYER_H_

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/util.h"

namespace xla {

// Transforms async collectives (e.g., an all-reduce-start and all-reduce-done)
// into synchronous collective (e.g., an all-reduce).
//
// Note that this transformation is similar to but different than the one in
// convert_async_collectives_to_sync.h. ConvertAsyncCollectivesToSync is run
// after scheduling and replaces start/done pairs that don't have any
// overlapping computation. AsyncCollectiveDestroyer is run before scheduling
// and removes start/done pairs based on the provided predicates.
class AsyncCollectiveDestroyer : public HloModulePass {
 public:
  struct Config {
    HloPredicate convert_all_reduce = HloPredicateFalse;
    HloPredicate convert_all_gather = HloPredicateFalse;
    HloPredicate convert_collective_broadcast = HloPredicateFalse;
    HloPredicate convert_collective_permute = HloPredicateFalse;
    HloPredicate convert_all_to_all = HloPredicateFalse;
    HloPredicate convert_reduce_scatter = HloPredicateFalse;
  };

  explicit AsyncCollectiveDestroyer(Config config)
      : config_(std::move(config)) {}

  absl::string_view name() const override {
    return "async-collective-destroyer";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  Config config_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_COLLECTIVES_ASYNC_COLLECTIVE_DESTROYER_H_

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

#include "xla/hlo/transforms/collectives/async_collective_destroyer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace {

namespace m = xla::testing::opcode_matchers;

class AsyncCollectiveDestroyerTest : public HloHardwareIndependentTestBase {
 public:
  absl::Status RunPass(HloModule* module, bool expect_change,
                       AsyncCollectiveDestroyer::Config config) {
    TF_ASSIGN_OR_RETURN(bool changed,
                        AsyncCollectiveDestroyer{config}.Run(module));
    EXPECT_EQ(changed, expect_change);
    VLOG(1) << module->ToString();
    return absl::OkStatus();
  }
};

// AllReduce tests
TEST_F(AsyncCollectiveDestroyerTest, AllReduceReplaced) {
  const absl::string_view hlo_string = R"(
      HloModule test

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start = u32[] all-reduce-start(id), to_apply=apply_op
        ROOT done = u32[] all-reduce-done(start)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  config.convert_all_reduce = HloPredicateTrue;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(), m::AllReduce());
}

TEST_F(AsyncCollectiveDestroyerTest, AllReduceNotReplaced) {
  const absl::string_view hlo_string = R"(
      HloModule test

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start = u32[] all-reduce-start(id), to_apply=apply_op
        ROOT done = u32[] all-reduce-done(start)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/false, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::AllReduceDone());
}

TEST_F(AsyncCollectiveDestroyerTest, AllReduceReplacedWithUnrelatedOp) {
  const absl::string_view hlo_string = R"(
      HloModule test

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        start = u32[] all-reduce-start(id), to_apply=apply_op
        unrelated = u32[] constant(42)
        ROOT done = u32[] all-reduce-done(start)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  config.convert_all_reduce = HloPredicateTrue;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(), m::AllReduce());
}

// AllGather tests
TEST_F(AsyncCollectiveDestroyerTest, AllGatherReplaced) {
  const absl::string_view hlo_string = R"(
      HloModule test

      ENTRY test_computation {
        p = u32[2] parameter(0)
        start = (u32[2], u32[4]) all-gather-start(p), dimensions={0}
        ROOT done = u32[4] all-gather-done(start)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  config.convert_all_gather = HloPredicateTrue;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(), m::AllGather());
}

TEST_F(AsyncCollectiveDestroyerTest, AllGatherNotReplaced) {
  const absl::string_view hlo_string = R"(
      HloModule test

      ENTRY test_computation {
        p = u32[2] parameter(0)
        start = (u32[2], u32[4]) all-gather-start(p), dimensions={0}
        ROOT done = u32[4] all-gather-done(start)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/false, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::AllGatherDone());
}

TEST_F(AsyncCollectiveDestroyerTest, AllGatherReplacedWithUnrelatedOp) {
  const absl::string_view hlo_string = R"(
      HloModule test

      ENTRY test_computation {
        p = u32[2] parameter(0)
        start = (u32[2], u32[4]) all-gather-start(p), dimensions={0}
        unrelated = u32[] constant(42)
        ROOT done = u32[4] all-gather-done(start)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  config.convert_all_gather = HloPredicateTrue;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(), m::AllGather());
}

// CollectivePermute tests
TEST_F(AsyncCollectiveDestroyerTest, CollectivePermuteReplaced) {
  const absl::string_view hlo_string = R"(
      HloModule test

      ENTRY test_computation {
        p = u32[2] parameter(0)
        start = (u32[2], u32[2], u32[], u32[]) collective-permute-start(p), source_target_pairs={{0,1}, {1,0}}
        ROOT done = u32[2] collective-permute-done(start)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  config.convert_collective_permute = HloPredicateTrue;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::CollectivePermute());
}

TEST_F(AsyncCollectiveDestroyerTest, CollectivePermuteNotReplaced) {
  const absl::string_view hlo_string = R"(
      HloModule test

      ENTRY test_computation {
        p = u32[2] parameter(0)
        start = (u32[2], u32[2], u32[], u32[]) collective-permute-start(p), source_target_pairs={{0,1}, {1,0}}
        ROOT done = u32[2] collective-permute-done(start)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/false, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::CollectivePermuteDone());
}

TEST_F(AsyncCollectiveDestroyerTest, CollectivePermuteReplacedWithUnrelatedOp) {
  const absl::string_view hlo_string = R"(
      HloModule test

      ENTRY test_computation {
        p = u32[2] parameter(0)
        start = (u32[2], u32[2], u32[], u32[]) collective-permute-start(p), source_target_pairs={{0,1}, {1,0}}
        unrelated = u32[] constant(42)
        ROOT done = u32[2] collective-permute-done(start)
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  config.convert_collective_permute = HloPredicateTrue;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::CollectivePermute());
}

// CollectiveBroadcast tests
TEST_F(AsyncCollectiveDestroyerTest, CollectiveBroadcastReplaced) {
  const absl::string_view hlo_string = R"(
      HloModule test

      broadcast_computation {
        p0 = u32[2] parameter(0)
        ROOT result = u32[2] collective-broadcast(p0), replica_groups={{0}}
      }

      ENTRY test_computation {
        p = u32[2] parameter(0)
        start = ((u32[2]), u32[2]) async-start(p), calls=broadcast_computation
        ROOT done = u32[2] async-done(start), calls=broadcast_computation
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  config.convert_collective_broadcast = HloPredicateTrue;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::CollectiveBroadcast());
}

TEST_F(AsyncCollectiveDestroyerTest, CollectiveBroadcastNotReplaced) {
  const absl::string_view hlo_string = R"(
      HloModule test

      broadcast_computation {
        p0 = u32[2] parameter(0)
        ROOT result = u32[2] collective-broadcast(p0), replica_groups={{0}}
      }

      ENTRY test_computation {
        p = u32[2] parameter(0)
        start = ((u32[2]), u32[2]) async-start(p), calls=broadcast_computation
        ROOT done = u32[2] async-done(start), calls=broadcast_computation
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/false, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(), m::AsyncDone());
}

TEST_F(AsyncCollectiveDestroyerTest,
       CollectiveBroadcastReplacedWithUnrelatedOp) {
  const absl::string_view hlo_string = R"(
      HloModule test

      broadcast_computation {
        p0 = u32[2] parameter(0)
        ROOT result = u32[2] collective-broadcast(p0), replica_groups={{0}}
      }

      ENTRY test_computation {
        p = u32[2] parameter(0)
        start = ((u32[2]), u32[2]) async-start(p), calls=broadcast_computation
        unrelated = u32[] constant(42)
        ROOT done = u32[2] async-done(start), calls=broadcast_computation
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  config.convert_collective_broadcast = HloPredicateTrue;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::CollectiveBroadcast());
}

// AllToAll tests
TEST_F(AsyncCollectiveDestroyerTest, AllToAllReplaced) {
  const absl::string_view hlo_string = R"(
      HloModule test

      all_to_all_computation {
        p0 = u32[2] parameter(0)
        ROOT result = u32[2] all-to-all(p0), replica_groups={{0,1}}, dimensions={0}
      }

      ENTRY test_computation {
        p = u32[2] parameter(0)
        start = ((u32[2]), u32[2]) async-start(p), calls=all_to_all_computation
        ROOT done = u32[2] async-done(start), calls=all_to_all_computation
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  config.convert_all_to_all = HloPredicateTrue;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(), m::AllToAll());
}

TEST_F(AsyncCollectiveDestroyerTest, AllToAllNotReplaced) {
  const absl::string_view hlo_string = R"(
      HloModule test

      all_to_all_computation {
        p0 = u32[2] parameter(0)
        ROOT result = u32[2] all-to-all(p0), replica_groups={{0,1}}, dimensions={0}
      }

      ENTRY test_computation {
        p = u32[2] parameter(0)
        start = ((u32[2]), u32[2]) async-start(p), calls=all_to_all_computation
        ROOT done = u32[2] async-done(start), calls=all_to_all_computation
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/false, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(), m::AsyncDone());
}

TEST_F(AsyncCollectiveDestroyerTest, AllToAllReplacedWithUnrelatedOp) {
  const absl::string_view hlo_string = R"(
      HloModule test

      all_to_all_computation {
        p0 = u32[2] parameter(0)
        ROOT result = u32[2] all-to-all(p0), replica_groups={{0,1}}, dimensions={0}
      }

      ENTRY test_computation {
        p = u32[2] parameter(0)
        start = ((u32[2]), u32[2]) async-start(p), calls=all_to_all_computation
        unrelated = u32[] constant(42)
        ROOT done = u32[2] async-done(start), calls=all_to_all_computation
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  config.convert_all_to_all = HloPredicateTrue;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(), m::AllToAll());
}

// ReduceScatter tests
TEST_F(AsyncCollectiveDestroyerTest, ReduceScatterReplaced) {
  const absl::string_view hlo_string = R"(
      HloModule test

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      reduce_scatter_computation {
        p0 = u32[4] parameter(0)
        ROOT result = u32[2] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0}, to_apply=apply_op
      }

      ENTRY test_computation {
        p = u32[4] parameter(0)
        start = ((u32[4]), u32[2]) async-start(p), calls=reduce_scatter_computation
        ROOT done = u32[2] async-done(start), calls=reduce_scatter_computation
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  config.convert_reduce_scatter = HloPredicateTrue;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::ReduceScatter());
}

TEST_F(AsyncCollectiveDestroyerTest, ReduceScatterNotReplaced) {
  const absl::string_view hlo_string = R"(
      HloModule test

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      reduce_scatter_computation {
        p0 = u32[4] parameter(0)
        ROOT result = u32[2] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0}, to_apply=apply_op
      }

      ENTRY test_computation {
        p = u32[4] parameter(0)
        start = ((u32[4]), u32[2]) async-start(p), calls=reduce_scatter_computation
        ROOT done = u32[2] async-done(start), calls=reduce_scatter_computation
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/false, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(), m::AsyncDone());
}

TEST_F(AsyncCollectiveDestroyerTest, ReduceScatterReplacedWithUnrelatedOp) {
  const absl::string_view hlo_string = R"(
      HloModule test

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      reduce_scatter_computation {
        p0 = u32[4] parameter(0)
        ROOT result = u32[2] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0}, to_apply=apply_op
      }

      ENTRY test_computation {
        p = u32[4] parameter(0)
        start = ((u32[4]), u32[2]) async-start(p), calls=reduce_scatter_computation
        unrelated = u32[] constant(42)
        ROOT done = u32[2] async-done(start), calls=reduce_scatter_computation
      }
    )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  AsyncCollectiveDestroyer::Config config;
  config.convert_reduce_scatter = HloPredicateTrue;
  TF_ASSERT_OK(RunPass(module.get(), /*expect_change=*/true, config));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              m::ReduceScatter());
}

}  // namespace
}  // namespace xla

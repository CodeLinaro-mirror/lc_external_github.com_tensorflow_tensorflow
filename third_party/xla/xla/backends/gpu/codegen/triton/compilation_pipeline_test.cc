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

#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_join.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"

namespace xla {
namespace gpu {
namespace {

TEST(CompilationPipelineTest, UnswitchLoopsAfterLICM) {
  // As the loop unswitcher relies on loop invariant code to be outside of the
  // loop, we need to check that LICM runs before the loop unswitcher.
  mlir::MLIRContext ctx;
  mlir::PassManager pm(&ctx);

  CreateTritonXlaPipeline(&pm, stream_executor::CudaComputeCapability(),
                          /*rewrite_int4=*/false, /*allow_tma=*/true,
                          /*convert_unsupported_types=*/true);

  std::vector<std::string> pass_names;
  for (const mlir::Pass& pass : pm.getPasses()) {
    pass_names.push_back(pass.getName().str());
  }
  auto licm_it = std::find(pass_names.begin(), pass_names.end(),
                           "LoopInvariantCodeMotion");
  auto unswitch_it = std::find(pass_names.begin(), pass_names.end(),
                               "TritonXLAUnswitchLoopsPass");

  ASSERT_NE(licm_it, pass_names.end())
      << "LoopInvariantCodeMotionPass not found";
  ASSERT_NE(unswitch_it, pass_names.end())
      << "TritonXLAUnswitchLoopsPass not found";

  int licm_index = std::distance(pass_names.begin(), licm_it);
  int unswitch_index = std::distance(pass_names.begin(), unswitch_it);
  // There is no hard requirement to run LICM **immediately** before the loop
  // unswitcher but you should consider if the newly added pass might interact
  // with the loop unswitcher.
  EXPECT_EQ(unswitch_index, licm_index + 1)
      << "TritonXLAUnswitchLoopsPass is expected to run right after "
         "LoopInvariantCodeMotionPass. Got passes: "
      << absl::StrJoin(pass_names, ", ");
}

}  // namespace
}  // namespace gpu
}  // namespace xla

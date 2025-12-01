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

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/cublaslt.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

using CublasLtBackendConfig = AutotuneResult::GemmKey;

class CublasLtBackendTest : public HloHardwareIndependentTestBase {
 protected:
  DebugOptions debug_options_;
  NVPTXCompiler compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  CublasLtBackend backend_;

  CublasLtBackendTest()
      : stream_executor_(PlatformUtil::GetDefaultPlatform()
                             .value()
                             ->ExecutorForDevice(0)
                             .value()),
        target_config_(stream_executor_),
        backend_(stream_executor_, &debug_options_, &compiler_,
                 &target_config_) {}

  CublasLtBackendConfig ExpectedDefaultAlgorithm() {
    auto config = AutotuneResult::GemmKey();
    config.set_algorithm(se::blas::kDefaultAlgorithm);
    return config;
  }
};

TEST_F(CublasLtBackendTest, CompileFp8SwapOperands) {
  if (!stream_executor_->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeast(8, 9)) {
    GTEST_SKIP() << "FP8 requires compute capability 8.9 or higher";
  }
  // CuBLASLt requires the operands to be in a specific layout (transposed /
  // non-transposed) for FP8 matrix multiplication. This HLO defines a
  // row-major output which forces the backend to swap operands, implicitly
  // satisfying the layout requirements.
  const char kFp8MatmulWithSwapHlo[] = R"(
  HloModule module

  ENTRY %main (lhs: f8e4m3fn[16,16], rhs: f8e4m3fn[16,16], lhs_scale: f32[], rhs_scale: f32[]) -> f32[16,16] {
    %lhs = f8e4m3fn[16,16]{1,0} parameter(0)
    %rhs = f8e4m3fn[16,16]{1,0} parameter(1)
    %lhs_scale = f32[] parameter(2)
    %rhs_scale = f32[] parameter(3)

    %custom-call = (f32[16,16]{1,0}, s8[100]{0}) custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__cublas$lt$matmul$f8",
      backend_config={"gemm_backend_config":{
        "dot_dimension_numbers":{
          "lhs_contracting_dimensions":["1"],
          "rhs_contracting_dimensions":["0"],
          "lhs_batch_dimensions":[],
          "rhs_batch_dimensions":[]
        },
        "alpha_real": 1,
        "beta": 0
      }}
    ROOT %get-tuple-element = f32[16,16]{1,0} get-tuple-element(%custom-call), index=0
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kFp8MatmulWithSwapHlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BackendConfig> config,
      backend_.GetDefaultConfig(
          *(module->entry_computation()->root_instruction()->operand(0))));
  absl::StatusOr<std::unique_ptr<Executable>> executable = backend_.Compile(
      *(module->entry_computation()->root_instruction()->operand(0)), *config);
  EXPECT_THAT(executable, absl_testing::IsOk());
}

}  // namespace gpu
}  // namespace xla

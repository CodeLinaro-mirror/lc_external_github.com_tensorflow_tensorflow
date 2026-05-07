/* Copyright 2024 The OpenXLA Authors.

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

#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/gpu_codegen_test.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/debug_options_flags.h"
#include "xla/error_spec.h"
#include "xla/literal.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::gpu {

HloModuleConfig GetModuleConfig() {
  HloModuleConfig config;
  config.set_debug_options(DefaultDebugOptionsIgnoringFlags());
  return config;
}

class FloatConversionTest : public HloPjRtGpuTestBase {
 public:
  void RunAndCompare(absl::string_view hlo_text, const ErrorSpec& error_spec) {
    HloModuleConfig module_config = GetModuleConfig();
    ASSERT_OK_AND_ASSIGN(auto module,
                         ParseAndReturnVerifiedModule(hlo_text, module_config));
    ASSERT_OK_AND_ASSIGN(auto reference_module,
                         ParseAndReturnVerifiedModule(hlo_text, module_config));
    EXPECT_TRUE(RunAndCompareTwoModules(
        std::move(module), std::move(reference_module), error_spec));
  }
};

class FloatConversionParamTest
    : public FloatConversionTest,
      public ::testing::WithParamInterface<std::string> {};

INSTANTIATE_TEST_SUITE_P(FloatConversionParamSuite, FloatConversionParamTest,
                         ::testing::Values("f64", "f32", "f16", "bf16",
                                           "f4e2m1fn", "f8e5m2", "f8e5m2fnuz",
                                           "f8e4m3", "f8e4m3fn", "f8e4m3fnuz",
                                           "f8e4m3b11fnuz", "f8e3m4",
                                           "f8e8m0fnu"));

TEST_P(FloatConversionParamTest, FloatToF16) {
  auto type_name = GetParam();
  RunAndCompare(absl::StrFormat(R"(ENTRY m {
                                    p0 = %s[] parameter(0)
                                    ROOT c1 = f16[] convert(p0)
                                  })",
                                type_name),
                ErrorSpec{1e-5, 1e-5});
}

TEST_P(FloatConversionParamTest, F16ToFloat) {
  auto type_name = GetParam();
  RunAndCompare(absl::StrFormat(R"(ENTRY m {
                                    p0 = f16[] parameter(0)
                                    ROOT c1 = %s[] convert(p0)
                                  })",
                                type_name),
                ErrorSpec{1e-5, 1e-5});
}

TEST_P(FloatConversionParamTest, FloatToF32) {
  auto type_name = GetParam();
  RunAndCompare(absl::StrFormat(R"(ENTRY m {
                                    p0 = %s[] parameter(0)
                                    ROOT c1 = f32[] convert(p0)
                                  })",
                                type_name),
                ErrorSpec{1e-5, 1e-5});
}

TEST_P(FloatConversionParamTest, F32ToFloat) {
  auto type_name = GetParam();
  RunAndCompare(absl::StrFormat(R"(ENTRY m {
                                    p0 = f32[] parameter(0)
                                    ROOT c1 = %s[] convert(p0)
                                  })",
                                type_name),
                ErrorSpec{1e-5, 1e-5});
}

TEST_F(FloatConversionTest, F32ToPred) {
  RunAndCompare(R"(ENTRY m {
                    iota = f32[1000] iota(), iota_dimension=0
                    c500 = f32[] constant(500)
                    c500_b = f32[1000] broadcast(c500), dimensions={}
                    sub = f32[1000] subtract(iota, c500_b)
                    ROOT c = pred[1000] convert(sub)
                  })",
                ErrorSpec{1e-5, 1e-5});

  RunAndCompare(R"(ENTRY m {
                      n = f32[] constant(nan)
                      ROOT c = pred[] convert(n)
                    })",
                ErrorSpec{1e-5, 1e-5});
}

TEST_F(FloatConversionTest, F32ToS8) {
  RunAndCompare(R"(ENTRY m {
                    iota = f32[1000] iota(), iota_dimension=0
                    c500 = f32[] constant(500)
                    c500_b = f32[1000] broadcast(c500), dimensions={}
                    sub = f32[1000] subtract(iota, c500_b)
                    ROOT c = s8[1000] convert(sub)
                  })",
                ErrorSpec{1e-5, 1e-5});

  RunAndCompare(R"(ENTRY m {
                    n = f32[] constant(nan)
                    ROOT c = s8[] convert(n)
                  })",
                ErrorSpec{1e-5, 1e-5});
}

TEST_F(FloatConversionTest, F32ToU8) {
  RunAndCompare(R"(ENTRY m {
                    iota = f32[1000] iota(), iota_dimension=0
                    c500 = f32[] constant(500)
                    c500_b = f32[1000] broadcast(c500), dimensions={}
                    sub = f32[1000] subtract(iota, c500_b)
                    ROOT c = u8[1000] convert(sub)
                  })",
                ErrorSpec{1e-5, 1e-5});

  RunAndCompare(R"(ENTRY m {
                    n = f32[] constant(nan)
                    ROOT c = u8[] convert(n)
                  })",
                ErrorSpec{1e-5, 1e-5});
}

TEST_F(FloatConversionTest, BF16ToS16IsBroken) {
  RunAndCompare(R"(ENTRY m {
                    iota = u16[65536] iota(), iota_dimension=0
                    bc = bf16[65536] bitcast-convert(iota)
                    ROOT c = s16[65536] convert(bc)
                  })",
                ErrorSpec{1e-5, 1e-5});
}

}  // namespace xla::gpu

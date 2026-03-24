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

#include "xla/hlo/transforms/simplifiers/recognize_reduce_window.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

class RecognizeReduceWindowTest : public HloHardwareIndependentTestBase {
 public:
  void CheckRecognizeReduceWindow(absl::string_view hlo,
                                  std::optional<absl::string_view> expected) {
    RunAndFilecheckHloRewrite(hlo, RecognizeReduceWindow{}, expected);
  }
};

TEST_F(RecognizeReduceWindowTest, NonMatchingOpcode) {
  const absl::string_view hlo_string = R"(
HloModule NonMatchingOpcode

ENTRY main {
  x = f32[10] parameter(0)
  slice_1 = f32[8] slice(x), slice={[0:8]}
  slice_2 = f32[8] slice(x), slice={[2:10]}
  ROOT div = f32[8] divide(slice_1, slice_2)
}
)";
  // Should not match because divide is not commutative.
  CheckRecognizeReduceWindow(hlo_string, std::nullopt);
}

TEST_F(RecognizeReduceWindowTest, UserExample) {
  const absl::string_view hlo_string = R"(
HloModule UserExample

ENTRY main {
  %concatenate.136 = f64[4,760,1534]{2,1,0} parameter(0)
  %slice.446 = f64[4,760,1528]{2,1,0} slice(%concatenate.136), slice={[0:4], [0:760], [3:1531]}
  %slice.447 = f64[4,760,1528]{2,1,0} slice(%concatenate.136), slice={[0:4], [0:760], [2:1530]}
  ROOT %add = f64[4,760,1528]{2,1,0} add(%slice.446, %slice.447)
}
)";
  const absl::string_view expected = R"(
// CHECK: reduce-window
  )";
  CheckRecognizeReduceWindow(hlo_string, expected);
}

TEST_F(RecognizeReduceWindowTest, AddTwoSlices) {
  const absl::string_view hlo_string = R"(
HloModule AddTwoSlices

ENTRY main {
  x = f32[10] parameter(0)
  slice_1 = f32[8] slice(x), slice={[0:8]}
  slice_2 = f32[8] slice(x), slice={[2:10]}
  ROOT add = f32[8] add(slice_1, slice_2)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: %[[REDUCER_NAME:.*]] ({{.*}}) -> f32[] {
// CHECK:   [[LHS:%.*]] = f32[] parameter(0)
// CHECK:   [[RHS:%.*]] = f32[] parameter(1)
// CHECK:   ROOT {{.*}} = f32[] add([[LHS]], [[RHS]])
// CHECK: }
// CHECK: ENTRY %main ({{.*}}: f32[10]) -> f32[8] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[SLICE:%.*]] = f32[10]{0} slice([[X]]), slice={[0:10]}
// CHECK:   [[ZERO:%.*]] = f32[] constant(0)
// CHECK:   ROOT {{.*}} = f32[8]{0} reduce-window([[SLICE]], [[ZERO]]), window={size=2 rhs_dilate=2}, to_apply=%[[REDUCER_NAME]]
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, AddTwoPads) {
  const absl::string_view hlo_string = R"(
HloModule AddTwoPads

ENTRY main {
  x = f32[10] parameter(0)
  pad_val = f32[] constant(0.0)
  pad_1 = f32[12] pad(x, pad_val), padding=2_0
  pad_2 = f32[12] pad(x, pad_val), padding=0_2
  ROOT add = f32[12] add(pad_1, pad_2)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: %[[REDUCER_NAME:.*]] ({{.*}}) -> f32[] {
// CHECK:   [[LHS:%.*]] = f32[] parameter(0)
// CHECK:   [[RHS:%.*]] = f32[] parameter(1)
// CHECK:   ROOT {{.*}} = f32[] add([[LHS]], [[RHS]])
// CHECK: }
// CHECK: ENTRY %main ({{.*}}: f32[10]) -> f32[12] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[PAD_VAL:%.*]] = f32[] constant(0)
// CHECK:   [[PAD_UNION:%.*]] = f32[14]{0} pad([[X]], [[PAD_VAL]]), padding=2_2
// CHECK:   [[ZERO:%.*]] = f32[] constant(0)
// CHECK:   ROOT {{.*}} = f32[12]{0} reduce-window([[PAD_UNION]], [[ZERO]]), window={size=2 rhs_dilate=2}, to_apply=%[[REDUCER_NAME]]
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, AddChainedPads) {
  const absl::string_view hlo_string = R"(
HloModule AddChainedPads

ENTRY main {
  x = f32[10] parameter(0)
  pad_val = f32[] constant(0.0)
  pad_1 = f32[14] pad(x, pad_val), padding=4_0
  pad_2 = f32[14] pad(x, pad_val), padding=2_2
  pad_3 = f32[14] pad(x, pad_val), padding=0_4
  add_1 = f32[14] add(pad_1, pad_2)
  ROOT add_2 = f32[14] add(add_1, pad_3)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: %[[REDUCER_NAME:.*]] ({{.*}}) -> f32[] {
// CHECK:   [[LHS:%.*]] = f32[] parameter(0)
// CHECK:   [[RHS:%.*]] = f32[] parameter(1)
// CHECK:   ROOT {{.*}} = f32[] add([[LHS]], [[RHS]])
// CHECK: }
// CHECK: ENTRY %main ({{.*}}: f32[10]) -> f32[14] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[PAD_VAL:%.*]] = f32[] constant(0)
// CHECK:   [[PAD_UNION:%.*]] = f32[18]{0} pad([[X]], [[PAD_VAL]]), padding=4_4
// CHECK:   [[ZERO:%.*]] = f32[] constant(0)
// CHECK:   ROOT {{.*}} = f32[14]{0} reduce-window([[PAD_UNION]], [[ZERO]]), window={size=3 rhs_dilate=2}, to_apply=%[[REDUCER_NAME]]
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, AddFourChainedSlices) {
  const absl::string_view hlo_string = R"(
HloModule AddFourChainedSlices

ENTRY main {
  x = f32[10] parameter(0)
  slice_1 = f32[4] slice(x), slice={[0:4]}
  slice_2 = f32[4] slice(x), slice={[2:6]}
  slice_3 = f32[4] slice(x), slice={[4:8]}
  slice_4 = f32[4] slice(x), slice={[6:10]}
  add_1 = f32[4] add(slice_1, slice_2)
  add_2 = f32[4] add(add_1, slice_3)
  ROOT add_3 = f32[4] add(add_2, slice_4)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: %[[REDUCER_NAME:.*]] ({{.*}}) -> f32[] {
// CHECK:   [[LHS:%.*]] = f32[] parameter(0)
// CHECK:   [[RHS:%.*]] = f32[] parameter(1)
// CHECK:   ROOT {{.*}} = f32[] add([[LHS]], [[RHS]])
// CHECK: }
// CHECK: ENTRY %main ({{.*}}: f32[10]) -> f32[4] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[SLICE:%.*]] = f32[10]{0} slice([[X]]), slice={[0:10]}
// CHECK:   [[ZERO:%.*]] = f32[] constant(0)
// CHECK:   ROOT {{.*}} = f32[4]{0} reduce-window([[SLICE]], [[ZERO]]), window={size=4 rhs_dilate=2}, to_apply=%[[REDUCER_NAME]]
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, AddChainedSlices) {
  const absl::string_view hlo_string = R"(
HloModule AddChainedSlices

ENTRY main {
  x = f32[10] parameter(0)
  slice_1 = f32[6] slice(x), slice={[0:6]}
  slice_2 = f32[6] slice(x), slice={[2:8]}
  slice_3 = f32[6] slice(x), slice={[4:10]}
  add_1 = f32[6] add(slice_1, slice_2)
  ROOT add_2 = f32[6] add(add_1, slice_3)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: %[[REDUCER_NAME:.*]] ({{.*}}) -> f32[] {
// CHECK:   [[LHS:%.*]] = f32[] parameter(0)
// CHECK:   [[RHS:%.*]] = f32[] parameter(1)
// CHECK:   ROOT {{.*}} = f32[] add([[LHS]], [[RHS]])
// CHECK: }
// CHECK: ENTRY %main ({{.*}}: f32[10]) -> f32[6] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[SLICE:%.*]] = f32[10]{0} slice([[X]]), slice={[0:10]}
// CHECK:   [[ZERO:%.*]] = f32[] constant(0)
// CHECK:   ROOT {{.*}} = f32[6]{0} reduce-window([[SLICE]], [[ZERO]]), window={size=3 rhs_dilate=2}, to_apply=%[[REDUCER_NAME]]
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, SubtractSlicesToDotGeneral) {
  const absl::string_view hlo_string = R"(
HloModule SubtractSlices

ENTRY main {
  x = f32[10] parameter(0)
  slice_1 = f32[8] slice(x), slice={[0:8]}
  slice_2 = f32[8] slice(x), slice={[2:10]}
  ROOT sub = f32[8] subtract(slice_1, slice_2)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: ENTRY %main ({{.*}}: f32[10]) -> f32[8] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[SLICE_1:%.*]] = f32[8]{0} slice([[X]]), slice={[0:8]}
// CHECK:   [[RESHAPE:%.*]] = f32[1,8]{1,0} reshape([[SLICE_1]])
// CHECK:   [[SLICE_2:%.*]] = f32[8]{0} slice([[X]]), slice={[2:10]}
// CHECK:   [[RESHAPE_1:%.*]] = f32[1,8]{1,0} reshape([[SLICE_2]])
// CHECK:   [[CONCAT:%.*]] = f32[2,8]{1,0} concatenate([[RESHAPE]], [[RESHAPE_1]]), dimensions={0}
// CHECK:   [[CONSTANT:%.*]] = f32[2]{0} constant({1, -1})
// CHECK:   ROOT {{.*}} = f32[8]{0} dot([[CONCAT]], [[CONSTANT]]), lhs_contracting_dims={0}, rhs_contracting_dims={0}
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, FlattenAndSortWeightedSlices) {
  const absl::string_view hlo_string = R"(
HloModule FlattenSlices

ENTRY main {
  x = f32[10] parameter(0)
  slice_0 = f32[8] slice(x), slice={[1:9]}
  slice_1 = f32[8] slice(x), slice={[0:8]}
  slice_2 = f32[8] slice(x), slice={[2:10]}
  sub_1 = f32[8] subtract(slice_1, slice_2)
  ROOT add = f32[8] add(sub_1, slice_0)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: ENTRY %main ({{.*}}: f32[10]) -> f32[8] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[SLICE_1:%.*]] = f32[8]{0} slice([[X]]), slice={[0:8]}
// CHECK:   [[RESHAPE_1:%.*]] = f32[1,8]{1,0} reshape([[SLICE_1]])
// CHECK:   [[SLICE_0:%.*]] = f32[8]{0} slice([[X]]), slice={[1:9]}
// CHECK:   [[RESHAPE_0:%.*]] = f32[1,8]{1,0} reshape([[SLICE_0]])
// CHECK:   [[SLICE_2:%.*]] = f32[8]{0} slice([[X]]), slice={[2:10]}
// CHECK:   [[RESHAPE_2:%.*]] = f32[1,8]{1,0} reshape([[SLICE_2]])
// CHECK:   [[CONCAT:%.*]] = f32[3,8]{1,0} concatenate([[RESHAPE_1]], [[RESHAPE_0]], [[RESHAPE_2]]), dimensions={0}
// CHECK:   [[CONSTANT:%.*]] = f32[3]{0} constant({1, 1, -1})
// CHECK:   ROOT {{.*}} = f32[8]{0} dot([[CONCAT]], [[CONSTANT]]), lhs_contracting_dims={0}, rhs_contracting_dims={0}
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, HeterogeneousConvolutionToDot) {
  const absl::string_view hlo_string = R"(
HloModule HeterogeneousConvolutionToDot

ENTRY main {
  x = f32[10] parameter(0)
  w1 = f32[10] parameter(1)
  x_1 = f32[8] slice(x), slice={[0:8]}
  w_1 = f32[8] slice(w1), slice={[0:8]}
  m_1 = f32[8] multiply(x_1, w_1)
  
  y = f32[10] parameter(2)
  w2 = f32[10] parameter(3)
  x_2 = f32[8] slice(y), slice={[2:10]}
  w_2 = f32[8] slice(w2), slice={[2:10]}
  m_2 = f32[8] multiply(x_2, w_2)
  
  ROOT add = f32[8] add(m_1, m_2)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: ENTRY %main ({{.*}}: f32[10], {{.*}}: f32[10], {{.*}}: f32[10], {{.*}}: f32[10]) -> f32[8] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[SLICE_X1:%.*]] = f32[8]{0} slice([[X]]), slice={[0:8]}
// CHECK:   [[RESHAPE_X1:%.*]] = f32[1,8]{1,0} reshape([[SLICE_X1]])
// CHECK:   [[Y:%.*]] = f32[10]{0} parameter(2)
// CHECK:   [[SLICE_Y2:%.*]] = f32[8]{0} slice([[Y]]), slice={[2:10]}
// CHECK:   [[RESHAPE_Y2:%.*]] = f32[1,8]{1,0} reshape([[SLICE_Y2]])
// CHECK:   [[CONCAT_XY:%.*]] = f32[2,8]{1,0} concatenate([[RESHAPE_X1]], [[RESHAPE_Y2]]), dimensions={0}
// CHECK:   [[W1:%.*]] = f32[10]{0} parameter(1)
// CHECK:   [[SLICE_W1:%.*]] = f32[8]{0} slice([[W1]]), slice={[0:8]}
// CHECK:   [[RESHAPE_W1:%.*]] = f32[1,8]{1,0} reshape([[SLICE_W1]])
// CHECK:   [[W2:%.*]] = f32[10]{0} parameter(3)
// CHECK:   [[SLICE_W2:%.*]] = f32[8]{0} slice([[W2]]), slice={[2:10]}
// CHECK:   [[RESHAPE_W2:%.*]] = f32[1,8]{1,0} reshape([[SLICE_W2]])
// CHECK:   [[CONCAT_W:%.*]] = f32[2,8]{1,0} concatenate([[RESHAPE_W1]], [[RESHAPE_W2]]), dimensions={0}
// CHECK:   ROOT {{.*}} = f32[8]{0} dot([[CONCAT_XY]], [[CONCAT_W]]), lhs_batch_dims={1}, lhs_contracting_dims={0}, rhs_batch_dims={1}, rhs_contracting_dims={0}
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, WeightedSlicesToDotGeneral) {
  const absl::string_view hlo_string = R"(
HloModule WeightedSlices

ENTRY main {
  x = f32[10] parameter(0)
  slice_1 = f32[8] slice(x), slice={[0:8]}
  c1 = f32[] constant(3)
  b1 = f32[8] broadcast(c1), dimensions={}
  m1 = f32[8] multiply(slice_1, b1)
  slice_2 = f32[8] slice(x), slice={[2:10]}
  c2 = f32[] constant(4)
  b2 = f32[8] broadcast(c2), dimensions={}
  m2 = f32[8] multiply(slice_2, b2)
  ROOT add = f32[8] add(m1, m2)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: ENTRY %main ({{.*}}: f32[10]) -> f32[8] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[SLICE_1:%.*]] = f32[8]{0} slice([[X]]), slice={[0:8]}
// CHECK:   [[RESHAPE_1:%.*]] = f32[1,8]{1,0} reshape([[SLICE_1]])
// CHECK:   [[SLICE_2:%.*]] = f32[8]{0} slice([[X]]), slice={[2:10]}
// CHECK:   [[RESHAPE_2:%.*]] = f32[1,8]{1,0} reshape([[SLICE_2]])
// CHECK:   [[CONCAT:%.*]] = f32[2,8]{1,0} concatenate([[RESHAPE_1]], [[RESHAPE_2]]), dimensions={0}
// CHECK:   [[CONSTANT:%.*]] = f32[2]{0} constant({3, 4})
// CHECK:   ROOT {{.*}} = f32[8]{0} dot([[CONCAT]], [[CONSTANT]]), lhs_contracting_dims={0}, rhs_contracting_dims={0}
// CHECK: }
)");
}
TEST_F(RecognizeReduceWindowTest, AddChainedDynamicSlices) {
  const absl::string_view hlo_string = R"(
HloModule AddChainedDynamicSlices

ENTRY main {
  x = f32[10] parameter(0)
  c_0 = s32[] constant(0)
  slice_1 = f32[6] dynamic-slice(x, c_0), dynamic_slice_sizes={6}
  c_2 = s32[] constant(2)
  slice_2 = f32[6] dynamic-slice(x, c_2), dynamic_slice_sizes={6}
  c_4 = s32[] constant(4)
  slice_3 = f32[6] dynamic-slice(x, c_4), dynamic_slice_sizes={6}
  add_1 = f32[6] add(slice_1, slice_2)
  ROOT add_2 = f32[6] add(add_1, slice_3)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: %[[REDUCER_NAME:.*]] ({{.*}}) -> f32[] {
// CHECK:   [[LHS:%.*]] = f32[] parameter(0)
// CHECK:   [[RHS:%.*]] = f32[] parameter(1)
// CHECK:   ROOT {{.*}} = f32[] add([[LHS]], [[RHS]])
// CHECK: }
// CHECK: ENTRY %main ({{.*}}: f32[10]) -> f32[6] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[SLICE:%.*]] = f32[10]{0} slice([[X]]), slice={[0:10]}
// CHECK:   [[ZERO:%.*]] = f32[] constant(0)
// CHECK:   ROOT {{.*}} = f32[6]{0} reduce-window([[SLICE]], [[ZERO]]), window={size=3 rhs_dilate=2}, to_apply=%[[REDUCER_NAME]]
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, DynamicSliceOutOfBoundsClamping) {
  const absl::string_view hlo_string = R"(
HloModule DynamicSliceOutOfBoundsClamping

ENTRY main {
  x = f32[10] parameter(0)
  c_neg = s32[] constant(-5)
  slice_1 = f32[6] dynamic-slice(x, c_neg), dynamic_slice_sizes={6}
  c_15 = s32[] constant(15)
  slice_2 = f32[6] dynamic-slice(x, c_15), dynamic_slice_sizes={6}
  ROOT add = f32[6] add(slice_1, slice_2)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: %[[REDUCER_NAME:.*]] ({{.*}}) -> f32[] {
// CHECK:   [[LHS:%.*]] = f32[] parameter(0)
// CHECK:   [[RHS:%.*]] = f32[] parameter(1)
// CHECK:   ROOT {{.*}} = f32[] add([[LHS]], [[RHS]])
// CHECK: }
// CHECK: ENTRY %main ({{.*}}: f32[10]) -> f32[6] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[SLICE:%.*]] = f32[10]{0} slice([[X]]), slice={[0:10]}
// CHECK:   [[ZERO:%.*]] = f32[] constant(0)
// CHECK:   ROOT {{.*}} = f32[6]{0} reduce-window([[SLICE]], [[ZERO]]), window={size=2 rhs_dilate=4}, to_apply=%[[REDUCER_NAME]]
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, ArrayWeightConvolution) {
  const absl::string_view hlo_string = R"(
HloModule ArrayWeightConvolution

ENTRY main {
  x = f32[10] parameter(0)
  w = f32[10] parameter(1)
  x_1 = f32[8] slice(x), slice={[0:8]}
  w_1 = f32[8] slice(w), slice={[0:8]}
  m_1 = f32[8] multiply(x_1, w_1)
  x_2 = f32[8] slice(x), slice={[2:10]}
  w_2 = f32[8] slice(w), slice={[2:10]}
  m_2 = f32[8] multiply(x_2, w_2)
  ROOT add = f32[8] add(m_1, m_2)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: ENTRY %main ({{.*}}: f32[10], {{.*}}: f32[10]) -> f32[8] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[SLICE_X1:%.*]] = f32[8]{0} slice([[X]]), slice={[0:8]}
// CHECK:   [[RESHAPE_X1:%.*]] = f32[1,8]{1,0} reshape([[SLICE_X1]])
// CHECK:   [[SLICE_X2:%.*]] = f32[8]{0} slice([[X]]), slice={[2:10]}
// CHECK:   [[RESHAPE_X2:%.*]] = f32[1,8]{1,0} reshape([[SLICE_X2]])
// CHECK:   [[CONCAT_X:%.*]] = f32[2,8]{1,0} concatenate([[RESHAPE_X1]], [[RESHAPE_X2]]), dimensions={0}
// CHECK:   [[W:%.*]] = f32[10]{0} parameter(1)
// CHECK:   [[SLICE_W1:%.*]] = f32[8]{0} slice([[W]]), slice={[0:8]}
// CHECK:   [[RESHAPE_W1:%.*]] = f32[1,8]{1,0} reshape([[SLICE_W1]])
// CHECK:   [[SLICE_W2:%.*]] = f32[8]{0} slice([[W]]), slice={[2:10]}
// CHECK:   [[RESHAPE_W2:%.*]] = f32[1,8]{1,0} reshape([[SLICE_W2]])
// CHECK:   [[CONCAT_W:%.*]] = f32[2,8]{1,0} concatenate([[RESHAPE_W1]], [[RESHAPE_W2]]), dimensions={0}
// CHECK:   ROOT {{.*}} = f32[8]{0} dot([[CONCAT_X]], [[CONCAT_W]]), lhs_batch_dims={1}, lhs_contracting_dims={0}, rhs_batch_dims={1}, rhs_contracting_dims={0}
// CHECK: }
)");
}

}  // namespace

TEST_F(RecognizeReduceWindowTest, UniversalDotGeneralFusion) {
  const absl::string_view hlo_string = R"(
HloModule UniversalDotGeneralFusion

ENTRY main {
  x = f32[10] parameter(0)
  w1 = f32[10] parameter(1)
  
  // Create non-slice matching geometry arrays
  x_1 = f32[10] reshape(x)
  w_1 = f32[10] reshape(w1)
  
  m_1 = f32[10] multiply(x_1, w_1)
  
  y = f32[10] parameter(2)
  w2 = f32[10] parameter(3)

  x_2 = f32[10] reshape(y)
  w_2 = f32[10] reshape(w2)

  m_2 = f32[10] multiply(x_2, w_2)
  
  ROOT add = f32[10] add(m_1, m_2)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: ENTRY %main ({{.*}}: f32[10], {{.*}}: f32[10], {{.*}}: f32[10], {{.*}}: f32[10]) -> f32[10] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[RESHAPE_X1:%.*]] = f32[10]{0} reshape([[X]])
// CHECK:   [[RESHAPE_X1_1:%.*]] = f32[1,10]{1,0} reshape([[RESHAPE_X1]])
// CHECK:   [[Y:%.*]] = f32[10]{0} parameter(2)
// CHECK:   [[RESHAPE_Y2:%.*]] = f32[10]{0} reshape([[Y]])
// CHECK:   [[RESHAPE_Y2_1:%.*]] = f32[1,10]{1,0} reshape([[RESHAPE_Y2]])
// CHECK:   [[CONCAT_XY:%.*]] = f32[2,10]{1,0} concatenate([[RESHAPE_X1_1]], [[RESHAPE_Y2_1]]), dimensions={0}
// CHECK:   [[W1:%.*]] = f32[10]{0} parameter(1)
// CHECK:   [[RESHAPE_W1:%.*]] = f32[10]{0} reshape([[W1]])
// CHECK:   [[RESHAPE_W1_1:%.*]] = f32[1,10]{1,0} reshape([[RESHAPE_W1]])
// CHECK:   [[W2:%.*]] = f32[10]{0} parameter(3)
// CHECK:   [[RESHAPE_W2:%.*]] = f32[10]{0} reshape([[W2]])
// CHECK:   [[RESHAPE_W2_1:%.*]] = f32[1,10]{1,0} reshape([[RESHAPE_W2]])
// CHECK:   [[CONCAT_W:%.*]] = f32[2,10]{1,0} concatenate([[RESHAPE_W1_1]], [[RESHAPE_W2_1]]), dimensions={0}
// CHECK:   ROOT {{.*}} = f32[10]{0} dot([[CONCAT_XY]], [[CONCAT_W]]), lhs_batch_dims={1}, lhs_contracting_dims={0}, rhs_batch_dims={1}, rhs_contracting_dims={0}
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, DeadParameterDoesNotCrash) {
  const absl::string_view hlo_string = R"(
HloModule DeadParameterDoesNotCrash

ENTRY main {
  x = f32[10] parameter(0)
  dead = f32[10] parameter(1)
  
  slice_1 = f32[8] slice(x), slice={[0:8]}
  slice_2 = f32[8] slice(x), slice={[2:10]}
  
  ROOT add = f32[8] add(slice_1, slice_2)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: %[[REDUCER_NAME:.*]] ({{.*}}) -> f32[] {
// CHECK:   [[LHS:%.*]] = f32[] parameter(0)
// CHECK:   [[RHS:%.*]] = f32[] parameter(1)
// CHECK:   ROOT {{.*}} = f32[] add([[LHS]], [[RHS]])
// CHECK: }
// CHECK: ENTRY %main ({{.*}}: f32[10], {{.*}}: f32[10]) -> f32[8] {
// CHECK-DAG:   [[DEAD:%.*]] = f32[10]{0} parameter(1)
// CHECK-DAG:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[SLICE:%.*]] = f32[10]{0} slice([[X]]), slice={[0:10]}
// CHECK:   [[ZERO:%.*]] = f32[] constant(0)
// CHECK:   ROOT {{.*}} = f32[8]{0} reduce-window([[SLICE]], [[ZERO]]), window={size=2 rhs_dilate=2}, to_apply=%[[REDUCER_NAME]]
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, XMinusX) {
  const absl::string_view hlo_string = R"(
HloModule XMinusX

ENTRY main {
  x = f32[10] parameter(0)
  ROOT res = f32[10] subtract(x, x)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: %main ({{.*}}) -> f32[10] {
// CHECK:   [[ZERO:%.*]] = f32[] constant(0)
// CHECK:   ROOT {{.*}} = f32[10]{0} broadcast([[ZERO]]), dimensions={}
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, ThreeXMinusTwoX) {
  // 3 * x + (-2) * x -> x
  const absl::string_view hlo_string = R"(
HloModule ThreeXMinusTwoX

ENTRY main {
  x = f32[10] parameter(0)
  p3 = f32[] constant(3)
  b3 = f32[10] broadcast(p3), dimensions={}
  m3 = f32[10] multiply(x, b3)
  
  p2 = f32[] constant(2)
  b2 = f32[10] broadcast(p2), dimensions={}
  m2 = f32[10] multiply(x, b2)
  
  ROOT res = f32[10] subtract(m3, m2)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: %main ({{.*}}) -> f32[10] {
// CHECK:   ROOT [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, XPlusX) {
  const absl::string_view hlo_string = R"(
HloModule XPlusX

ENTRY main {
  x = f32[10] parameter(0)
  ROOT res = f32[10] add(x, x)
}
)";
  CheckRecognizeReduceWindow(hlo_string, R"(
// CHECK: %main ({{.*}}) -> f32[10] {
// CHECK:   [[X:%.*]] = f32[10]{0} parameter(0)
// CHECK:   [[TWO:%.*]] = f32[] constant(2)
// CHECK:   [[BTWO:%.*]] = f32[10]{0} broadcast([[TWO]]), dimensions={}
// CHECK:   ROOT {{.*}} = f32[10]{0} multiply([[X]], [[BTWO]])
// CHECK: }
)");
}

TEST_F(RecognizeReduceWindowTest, XPlusY) {
  const absl::string_view hlo_string = R"(
HloModule XPlusY

ENTRY main {
  x = f32[10] parameter(0)
  y = f32[10] parameter(1)
  ROOT res = f32[10] add(x, y)
}
)";
  CheckRecognizeReduceWindow(hlo_string, std::nullopt);
}

TEST_F(RecognizeReduceWindowTest, XMinusY) {
  const absl::string_view hlo_string = R"(
HloModule XMinusY

ENTRY main {
  x = f32[10] parameter(0)
  y = f32[10] parameter(1)
  ROOT res = f32[10] subtract(x, y)
}
)";
  CheckRecognizeReduceWindow(hlo_string, std::nullopt);
}

TEST_F(RecognizeReduceWindowTest, YPlusX) {
  const absl::string_view hlo_string = R"(
HloModule YPlusX

ENTRY main {
  x = f32[10] parameter(0)
  y = f32[10] parameter(1)
  ROOT res = f32[10] add(y, x)
}
)";
  CheckRecognizeReduceWindow(hlo_string, std::nullopt);
}

TEST_F(RecognizeReduceWindowTest, YMinusX) {
  const absl::string_view hlo_string = R"(
HloModule YMinusX

ENTRY main {
  x = f32[10] parameter(0)
  y = f32[10] parameter(1)
  ROOT res = f32[10] subtract(y, x)
}
)";
  CheckRecognizeReduceWindow(hlo_string, std::nullopt);
}

TEST_F(RecognizeReduceWindowTest, ScalarAdd) {
  const absl::string_view hlo_string = R"(
HloModule ScalarAdd

ENTRY main {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT res = f32[] add(x, y)
}
)";
  CheckRecognizeReduceWindow(hlo_string, std::nullopt);
}

TEST_F(RecognizeReduceWindowTest, ScalarXPlusX) {
  const absl::string_view hlo_string = R"(
HloModule ScalarXPlusX

ENTRY main {
  x = f32[] parameter(0)
  ROOT res = f32[] add(x, x)
}
)";
  CheckRecognizeReduceWindow(hlo_string, std::nullopt);
}

}  // namespace xla

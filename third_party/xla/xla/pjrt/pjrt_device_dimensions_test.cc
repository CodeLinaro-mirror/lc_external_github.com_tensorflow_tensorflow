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

#include "xla/pjrt/pjrt_device_dimensions.h"

#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "xla/pjrt/proto/pjrt_device_dimensions.pb.h"

namespace xla {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::status::IsOkAndHolds;

TEST(PjRtDeviceDimensionsTest, Equality) {
  EXPECT_EQ((PjRtDeviceDimensions{1, 2, 3}), (PjRtDeviceDimensions{1, 2, 3}));
  EXPECT_NE((PjRtDeviceDimensions{1, 2, 3}), (PjRtDeviceDimensions{1, 2, 4}));
}

TEST(PjRtDeviceDimensionsTest, Ostream) {
  std::stringstream ss;
  ss << PjRtDeviceDimensions{1, 2, 3};
  EXPECT_EQ(ss.str(), "1,2,3");
}

TEST(PjRtDeviceDimensionsTest, AbslHashValue) {
  absl::flat_hash_set<PjRtDeviceDimensions> hash_set;
  hash_set.insert({1, 2, 3});
  hash_set.insert({0, 0, 0});
  hash_set.insert({1, 2, 3});  // Inserting again should not change size

  EXPECT_EQ(hash_set.size(), 2);
  EXPECT_TRUE(hash_set.contains({1, 2, 3}));
  EXPECT_TRUE(hash_set.contains({0, 0, 0}));
  EXPECT_FALSE(hash_set.contains({1, 2, 4}));
}

TEST(PjRtDeviceDimensionsTest, FromProto) {
  PjRtDeviceDimensionsProto proto;
  proto.set_x(1);
  proto.set_y(2);
  proto.set_z(3);
  EXPECT_THAT(PjRtDeviceDimensions::FromProto(proto),
              IsOkAndHolds(Eq(PjRtDeviceDimensions{1, 2, 3})));
}

TEST(PjRtDeviceDimensionsTest, ToProto) {
  PjRtDeviceDimensions bounds = {1, 2, 3};
  PjRtDeviceDimensionsProto proto = bounds.ToProto();
  EXPECT_EQ(proto.x(), 1);
  EXPECT_EQ(proto.y(), 2);
  EXPECT_EQ(proto.z(), 3);
}

TEST(AbslParseFlagTest, ValidInputs) {
  PjRtDeviceDimensions bounds;
  std::string err;

  EXPECT_TRUE(AbslParseFlag("1,2,3", &bounds, &err));
  EXPECT_EQ(bounds, (PjRtDeviceDimensions{1, 2, 3}));
  EXPECT_EQ(err, "");

  EXPECT_TRUE(AbslParseFlag("0,0,0", &bounds, &err));
  EXPECT_EQ(bounds, (PjRtDeviceDimensions{0, 0, 0}));
  EXPECT_EQ(err, "");
}

TEST(AbslParseFlagTest, InvalidInputs) {
  PjRtDeviceDimensions bounds;
  std::string err;

  EXPECT_FALSE(AbslParseFlag("1,2", &bounds, &err));
  EXPECT_THAT(err, HasSubstr("Incorrect number of elements"));

  EXPECT_FALSE(AbslParseFlag("1,2,3,4", &bounds, &err));
  EXPECT_THAT(err, HasSubstr("Incorrect number of elements"));

  EXPECT_FALSE(AbslParseFlag("", &bounds, &err));
  EXPECT_THAT(err, HasSubstr("Incorrect number of elements"));

  EXPECT_FALSE(AbslParseFlag("1,a,3", &bounds, &err));
  EXPECT_THAT(err, HasSubstr("Number parsing error"));

  EXPECT_FALSE(AbslParseFlag("1,2.5,3", &bounds, &err));
  EXPECT_THAT(err, HasSubstr("Number parsing error"));
}

TEST(AbslUnparseFlagTest, ConvertsCorrectly) {
  EXPECT_EQ(AbslUnparseFlag(PjRtDeviceDimensions{1, 2, 3}), "1,2,3");
  EXPECT_EQ(AbslUnparseFlag(PjRtDeviceDimensions{0, 0, 0}), "0,0,0");
}

}  // namespace
}  // namespace xla

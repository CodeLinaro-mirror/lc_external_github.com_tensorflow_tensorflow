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

#include "xla/python/ifrt/user_context_status_util.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/user_context_registry.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_to_from_proto.h"

namespace xla {
namespace ifrt {
namespace {

constexpr absl::string_view kIfrtUserContextPayloadUrl =
    "type.googleapis.com/ifrt.UserContext";

class TestUserContext : public llvm::RTTIExtends<TestUserContext, UserContext> {
 public:
  static UserContextRef Create(uint64_t id) {
    return tsl::TakeRef<TestUserContext>(new TestUserContext(id));
  }

  uint64_t Fingerprint() const override { return id_; }

  std::string DebugString() const override {
    return absl::StrCat("user context ", id_);
  }

  // No new `ID` is not defined because tests below do not exercise RTTI.

 private:
  explicit TestUserContext(uint64_t id) : id_(id) {}

  uint64_t id_;
};

TEST(UserContextStatusUtilTest, AttachUserContextId) {
  absl::Status status = absl::InvalidArgumentError("test");

  absl::Status new_status = AttachUserContextId(status, 100);
  {
    EXPECT_EQ(new_status.code(), status.code());
    EXPECT_EQ(new_status.message(), status.message());
    std::optional<absl::Cord> payload =
        new_status.GetPayload(kIfrtUserContextPayloadUrl);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->Flatten(), "100");
  }

  new_status = AttachUserContextId(new_status, 200);
  {
    EXPECT_EQ(new_status.code(), status.code());
    EXPECT_EQ(new_status.message(), status.message());
    std::optional<absl::Cord> payload =
        new_status.GetPayload(kIfrtUserContextPayloadUrl);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->Flatten(), "200,100");
  }
}

TEST(UserContextStatusUtilTest, NoOpToAttachUserContextIdToOkStatus) {
  absl::Status status;
  absl::Status new_status = AttachUserContextId(status, 100);
  TF_EXPECT_OK(new_status);
  std::optional<absl::Cord> payload =
      new_status.GetPayload(kIfrtUserContextPayloadUrl);
  ASSERT_FALSE(payload.has_value());
}

TEST(UserContextStatusUtilTest, AttachUserContextRef) {
  TrackedUserContextRef tracked_user_context1 =
      UserContextRegistry::Get().Register(TestUserContext::Create(100));
  TrackedUserContextRef tracked_user_context2 =
      UserContextRegistry::Get().Register(TestUserContext::Create(200));
  EXPECT_EQ(tracked_user_context1.use_count(), 1);
  EXPECT_EQ(tracked_user_context2.use_count(), 1);

  absl::Status status = absl::InvalidArgumentError("test");

  absl::Status new_status =
      AttachUserContextRef(status, tracked_user_context1->user_context());
  {
    EXPECT_EQ(new_status.code(), status.code());
    EXPECT_EQ(new_status.message(), status.message());
    std::optional<absl::Cord> payload =
        new_status.GetPayload(kIfrtUserContextPayloadUrl);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->Flatten(), "100");
    EXPECT_EQ(tracked_user_context1.use_count(), 2);
    EXPECT_EQ(tracked_user_context2.use_count(), 1);
  }

  new_status =
      AttachUserContextRef(new_status, tracked_user_context2->user_context());
  {
    EXPECT_EQ(new_status.code(), status.code());
    EXPECT_EQ(new_status.message(), status.message());
    std::optional<absl::Cord> payload =
        new_status.GetPayload(kIfrtUserContextPayloadUrl);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->Flatten(), "200,100");
    EXPECT_EQ(tracked_user_context1.use_count(), 2);
    EXPECT_EQ(tracked_user_context2.use_count(), 2);
  }
}

TEST(UserContextStatusUtilTest, NoOpToAttachUserContextRefToOkStatus) {
  absl::Status status;
  absl::Status new_status =
      AttachUserContextRef(status, TestUserContext::Create(100));
  TF_EXPECT_OK(new_status);
  std::optional<absl::Cord> payload =
      new_status.GetPayload(kIfrtUserContextPayloadUrl);
  ASSERT_FALSE(payload.has_value());
}

TEST(UserContextStatusUtilTest,
     ReattachUserContextRefsWithoutLiveUserContextRefs) {
  absl::Status status = absl::InvalidArgumentError("test");
  status = AttachUserContextId(std::move(status), 100);

  status = ReattachUserContextRefs(std::move(status));
  std::optional<absl::Cord> payload =
      status.GetPayload(kIfrtUserContextPayloadUrl);
  ASSERT_TRUE(payload.has_value());
  EXPECT_EQ(payload->Flatten(), "100");
}

TEST(UserContextStatusUtilTest,
     ReattachUserContextRefsWithLiveUserContextRefs) {
  absl::Status status = absl::InvalidArgumentError("test");
  status = AttachUserContextId(std::move(status), 100);

  TrackedUserContextRef tracked_user_context =
      UserContextRegistry::Get().Register(TestUserContext::Create(100));
  EXPECT_EQ(tracked_user_context.use_count(), 1);

  status = ReattachUserContextRefs(std::move(status));
  {
    std::optional<absl::Cord> payload =
        status.GetPayload(kIfrtUserContextPayloadUrl);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->Flatten(), "100");
    EXPECT_EQ(tracked_user_context.use_count(), 2);
  }

  status = ReattachUserContextRefs(std::move(status));
  {
    std::optional<absl::Cord> payload =
        status.GetPayload(kIfrtUserContextPayloadUrl);
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(payload->Flatten(), "100");
    // The use count can be greater than 2 depending on the implementation of
    // `ReattachUserContextRefs()` but never less than 2.
    EXPECT_GE(tracked_user_context.use_count(), 2);
  }
}

TEST(UserContextStatusUtilTest,
     NoOpToReattachUserContextRefsWithLiveUserContextRefsToOkStatus) {
  absl::Status status;
  absl::Status new_status = ReattachUserContextRefs(std::move(status));
  TF_EXPECT_OK(new_status);
  std::optional<absl::Cord> payload =
      new_status.GetPayload(kIfrtUserContextPayloadUrl);
  ASSERT_FALSE(payload.has_value());
}

TEST(UserContextStatusUtilTest, ExpandUserContexts) {
  absl::Status status = absl::InvalidArgumentError("test");
  status = AttachUserContextId(std::move(status), 100);
  status = AttachUserContextId(std::move(status), 200);

  {
    absl::Status expanded_status = ExpandUserContexts(status);
    EXPECT_EQ(
        expanded_status.message(),
        "test\n\t\n(failed to find a user context for ID: 200)\n\t\n(failed "
        "to find a user context for ID: 100)");
    std::optional<absl::Cord> payload =
        expanded_status.GetPayload(kIfrtUserContextPayloadUrl);
    EXPECT_FALSE(payload.has_value());
  }

  {
    TrackedUserContextRef tracked_user_context =
        UserContextRegistry::Get().Register(TestUserContext::Create(100));
    status = ReattachUserContextRefs(std::move(status));
  }
  {
    absl::Status expanded_status = ExpandUserContexts(status);
    EXPECT_EQ(expanded_status.message(),
              "test\n\t\n(failed to find a user context for ID: 200)\n\t\nuser "
              "context 100");
    std::optional<absl::Cord> payload =
        expanded_status.GetPayload(kIfrtUserContextPayloadUrl);
    EXPECT_FALSE(payload.has_value());
  }
}

TEST(UserContextStatusUtilTest, RoundtripPreserveUserContextIds) {
  absl::Status status = absl::InvalidArgumentError("test");
  status =
      AttachUserContextRef(std::move(status), TestUserContext::Create(100));
  status = AttachUserContextId(std::move(status), 200);
  {
    absl::Status expanded_status = ExpandUserContexts(status);
    EXPECT_EQ(expanded_status.message(),
              "test\n\t\n(failed to find a user context for ID: 200)\n\t\nuser "
              "context 100");
  }

  tensorflow::StatusProto status_proto = tsl::StatusToProto(status);
  status = tsl::StatusFromProto(status_proto);
  {
    absl::Status expanded_status = ExpandUserContexts(status);
    EXPECT_EQ(
        expanded_status.message(),
        "test\n\t\n(failed to find a user context for ID: 200)\n\t\n(failed "
        "to find a user context for ID: 100)");
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla

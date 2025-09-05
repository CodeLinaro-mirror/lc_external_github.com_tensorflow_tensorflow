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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/user_context_registry.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace ifrt {

namespace {

constexpr absl::string_view kIfrtUserContextPayloadUrl =
    "type.googleapis.com/ifrt.UserContext";

// Sets the payload of `status` with `user_context_ids_str` and
// `tracked_user_contexts`. If `old_payload` is provided, it will be captured in
// the new payload so that any references that were previously attached are
// transitively kept alive.
void SetPayloadWithUserContext(
    absl::Status& status, std::string user_context_ids_str,
    absl::InlinedVector<TrackedUserContextRef, 1> tracked_user_contexts,
    std::optional<absl::Cord> old_payload) {
  // Always create a new `Cord` as the payload using
  // `absl::MakeCordFromExternal()`. Using `Cord::Append()` can copy the newly
  // appended `Cord` because the length of the new `Cord` is short and will be
  // inlined with a copy, so we cannot mutate the old payload to construct the
  // new payload.
  //
  // The main cost is that the `user_context_ids_str_view` repeats any IDs seen
  // before, and thus its total size is quadratic. This is not a practical
  // problem because the number of `UserContextRef` attached to a status is
  // expected to be small. If this becomes a problem, the caller can use
  // `ReattachUserContextRefs()` that does not use the old payload.
  auto user_context_ids_holder =
      std::make_unique<std::string>(std::move(user_context_ids_str));
  absl::string_view user_context_ids_str_view = *user_context_ids_holder;
  status.SetPayload(
      kIfrtUserContextPayloadUrl,
      absl::MakeCordFromExternal(
          user_context_ids_str_view,
          [user_context_ids_holder = std::move(user_context_ids_holder),
           tracked_user_contexts = std::move(tracked_user_contexts),
           old_payload = std::move(old_payload)]() {}));
}

}  // namespace

absl::Status AttachUserContextId(absl::Status status,
                                 uint64_t user_context_id) {
  if (status.ok()) {
    return status;
  }
  std::optional<absl::Cord> payload =
      status.GetPayload(kIfrtUserContextPayloadUrl);
  std::string user_context_ids_str;
  if (payload.has_value()) {
    user_context_ids_str =
        absl::StrCat(user_context_id, ",", payload->Flatten());
  } else {
    user_context_ids_str = absl::StrCat(user_context_id);
  }
  SetPayloadWithUserContext(status, std::move(user_context_ids_str), {},
                            std::move(payload));
  return status;
}

absl::Status AttachUserContextRef(absl::Status status,
                                  UserContextRef user_context) {
  if (status.ok()) {
    return status;
  }
  TrackedUserContextRef tracked_user_context =
      UserContextRegistry::Get().Register(std::move(user_context));
  std::optional<absl::Cord> payload =
      status.GetPayload(kIfrtUserContextPayloadUrl);
  std::string user_context_ids_str;
  if (payload.has_value()) {
    user_context_ids_str =
        absl::StrCat(tracked_user_context->user_context()->Fingerprint(), ",",
                     payload->Flatten());
  } else {
    user_context_ids_str =
        absl::StrCat(tracked_user_context->user_context()->Fingerprint());
  }
  SetPayloadWithUserContext(status, std::move(user_context_ids_str),
                            {std::move(tracked_user_context)},
                            std::move(payload));
  return status;
}

absl::Status ReattachUserContextRefs(absl::Status status) {
  if (status.ok()) {
    return status;
  }
  std::optional<absl::Cord> payload =
      status.GetPayload(kIfrtUserContextPayloadUrl);
  if (!payload.has_value()) {
    return status;
  }

  std::string user_context_ids_str(payload->Flatten());
  std::vector<absl::string_view> split_user_context_id_strs =
      absl::StrSplit(user_context_ids_str, ',');
  absl::InlinedVector<TrackedUserContextRef, 1> tracked_user_contexts;
  tracked_user_contexts.reserve(split_user_context_id_strs.size());
  for (absl::string_view user_context_id_str : split_user_context_id_strs) {
    uint64_t user_context_id;
    if (!absl::SimpleAtoi(user_context_id_str, &user_context_id)) {
      continue;
    }
    TrackedUserContextRef tracked_user_context =
        UserContextRegistry::Get().Lookup(user_context_id);
    if (tracked_user_context == nullptr) {
      continue;
    }
    tracked_user_contexts.push_back(std::move(tracked_user_context));
  }
  // Note that the old payload is not preserved because `tracked_user_contexts`
  // will contain all live `UserContextRef`s that need to be kept alive.
  SetPayloadWithUserContext(status, std::move(user_context_ids_str),
                            std::move(tracked_user_contexts),
                            /*old_payload=*/std::nullopt);
  return status;
}

absl::Status ExpandUserContexts(absl::Status status) {
  if (status.ok()) {
    return status;
  }
  std::optional<absl::Cord> payload =
      status.GetPayload(kIfrtUserContextPayloadUrl);
  if (!payload.has_value()) {
    return status;
  }

  status.ErasePayload(kIfrtUserContextPayloadUrl);

  for (absl::string_view user_context_id_str :
       absl::StrSplit(payload->Flatten(), ',')) {
    uint64_t user_context_id;
    if (!absl::SimpleAtoi(user_context_id_str, &user_context_id)) {
      tsl::errors::AppendToMessage(
          &status,
          "\n(failed to parse a user context ID: ", user_context_id_str, ")");
      continue;
    }
    TrackedUserContextRef user_context =
        UserContextRegistry::Get().Lookup(user_context_id);
    if (user_context == nullptr) {
      tsl::errors::AppendToMessage(
          &status, "\n(failed to find a user context for ID: ", user_context_id,
          ")");
      continue;
    }
    tsl::errors::AppendToMessage(&status, "\n",
                                 user_context->user_context()->DebugString());
  }
  return status;
}

}  // namespace ifrt
}  // namespace xla

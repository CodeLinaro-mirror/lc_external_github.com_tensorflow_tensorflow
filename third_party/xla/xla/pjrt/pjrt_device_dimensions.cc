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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace xla {

std::string PjRtDeviceDimensions::ToString(absl::string_view sep) const {
  return absl::StrCat(x, sep, y, sep, z);
}

absl::StatusOr<PjRtDeviceDimensions> PjRtDeviceDimensions::FromString(
    absl::string_view text) {
  std::vector<std::string> bounds_str = absl::StrSplit(text, ',');

  if (bounds_str.size() != 3) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Incorrect number of elements specified for pjrt device "
        "dimensions %s, expected 3 got %d.",
        text, bounds_str.size()));
  }

  std::vector<int32_t> bounds_vec;
  for (auto const& b : bounds_str) {
    int32_t bound;
    if (absl::SimpleAtoi(b, &bound)) {
      bounds_vec.push_back(bound);
    } else {
      return absl::InvalidArgumentError(
          absl::StrFormat("Number parsing error for pjrt device dimensions %s "
                          "while parsing %s.",
                          text, b));
    }
  }

  return PjRtDeviceDimensions{bounds_vec[0], bounds_vec[1], bounds_vec[2]};
}

bool AbslParseFlag(absl::string_view text, PjRtDeviceDimensions* bounds,
                   std::string* err) {
  const auto status_or_dimensions = PjRtDeviceDimensions::FromString(text);
  if (!status_or_dimensions.ok()) {
    *err = status_or_dimensions.status().ToString();
    return false;
  }
  *bounds = status_or_dimensions.value();
  return true;
}

std::string AbslUnparseFlag(PjRtDeviceDimensions bounds) {
  return bounds.ToString();
}

}  // namespace xla

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

#ifndef XLA_PJRT_PJRT_DEVICE_DIMENSIONS_H_
#define XLA_PJRT_PJRT_DEVICE_DIMENSIONS_H_

#include <cstdint>
#include <ostream>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/proto/pjrt_device_dimensions.pb.h"

namespace xla {

// For anything where the x, y, z axes are integers, such as mesh bounds or
// chip coordinates.
struct PjRtDeviceDimensions {
  int32_t x;
  int32_t y;
  int32_t z;

  friend bool operator==(const PjRtDeviceDimensions& a,
                         const PjRtDeviceDimensions& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const PjRtDeviceDimensions& d) {
    return os << d.ToString();
  }

  template <typename H>
  friend H AbslHashValue(H h, const PjRtDeviceDimensions& c) {
    return H::combine(std::move(h), c.x, c.y, c.z);
  }

  static absl::StatusOr<PjRtDeviceDimensions> FromProto(
      const PjRtDeviceDimensionsProto& proto) {
    return PjRtDeviceDimensions{proto.x(), proto.y(), proto.z()};
  }

  PjRtDeviceDimensionsProto ToProto() const {
    PjRtDeviceDimensionsProto proto;
    proto.set_x(x);
    proto.set_y(y);
    proto.set_z(z);
    return proto;
  }

  std::string ToString(absl::string_view sep = ",") const;

  static absl::StatusOr<PjRtDeviceDimensions> FromString(
      absl::string_view text);
};

// Support for absl flags.
bool AbslParseFlag(absl::string_view text, PjRtDeviceDimensions* bounds,
                   std::string* err);
std::string AbslUnparseFlag(PjRtDeviceDimensions bounds);

}  // namespace xla

#endif  // XLA_PJRT_PJRT_DEVICE_DIMENSIONS_H_

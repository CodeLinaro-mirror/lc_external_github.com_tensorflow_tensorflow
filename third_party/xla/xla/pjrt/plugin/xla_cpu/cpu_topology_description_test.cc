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

#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::EqualsProto;

constexpr const uint64_t kDummyCpuPlatformId = 123456789;

TEST(CpuTopologyDescriptionTest, ToProto) {
  std::vector<CpuTopology::CpuDevice> cpu_devices = {{0, 0}, {0, 1}};
  std::vector<std::string> machine_attributes = {"attr1", "attr2"};
  CpuTopologyDescription topology(kDummyCpuPlatformId, "cpu", "1.0",
                                  cpu_devices, machine_attributes);

  TF_ASSERT_OK_AND_ASSIGN(PjRtTopologyDescriptionProto proto,
                          topology.ToProto());

  EXPECT_EQ(proto.platform_id(), kDummyCpuPlatformId);
  EXPECT_EQ(proto.platform_name(), "cpu");
  EXPECT_EQ(proto.platform_version(), "1.0");

  CpuTopologyProto cpu_topology_proto;
  ASSERT_TRUE(proto.platform_specific_topology().UnpackTo(&cpu_topology_proto));

  EXPECT_THAT(cpu_topology_proto, EqualsProto(R"pb(
                cpu_devices { process_index: 0 local_hardware_id: 0 }
                cpu_devices { process_index: 0 local_hardware_id: 1 }
                machine_attributes: "attr1"
                machine_attributes: "attr2"
              )pb"));
}

TEST(CpuTopologyDescriptionTest, FromProto) {
  PjRtTopologyDescriptionProto proto;
  proto.set_platform_id(kDummyCpuPlatformId);
  proto.set_platform_name("cpu");
  proto.set_platform_version("2.0");

  CpuTopologyProto cpu_topology_proto;
  auto* device1 = cpu_topology_proto.add_cpu_devices();
  device1->set_process_index(0);
  device1->set_local_hardware_id(0);
  auto* device2 = cpu_topology_proto.add_cpu_devices();
  device2->set_process_index(1);
  device2->set_local_hardware_id(1);
  cpu_topology_proto.add_machine_attributes("attrA");
  cpu_topology_proto.add_machine_attributes("attrB");

  proto.mutable_platform_specific_topology()->PackFrom(cpu_topology_proto);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtTopologyDescription> topology_desc,
      CpuTopologyDescription::FromProto(proto));

  CpuTopologyDescription* cpu_topology =
      dynamic_cast<CpuTopologyDescription*>(topology_desc.get());
  ASSERT_NE(cpu_topology, nullptr);

  EXPECT_EQ(cpu_topology->platform_id(), kDummyCpuPlatformId);
  EXPECT_EQ(cpu_topology->platform_name(), "cpu");
  EXPECT_EQ(cpu_topology->platform_version(), "2.0");

  const auto& devices = cpu_topology->cpu_topology().devices();
  ASSERT_EQ(devices.size(), 2);
  EXPECT_EQ(devices[0].process_id, 0);
  EXPECT_EQ(devices[0].local_device_id, 0);
  EXPECT_EQ(devices[1].process_id, 1);
  EXPECT_EQ(devices[1].local_device_id, 1);

  EXPECT_THAT(cpu_topology->cpu_topology().machine_attributes(),
              ElementsAre("attrA", "attrB"));
}

}  // namespace
}  // namespace xla

/* Copyright 2026 The OpenXLA Authors.

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

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_command_buffer.h"
#include "xla/stream_executor/cuda/cuda_kernel.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/cuda/cuda_stream.h"
#include "xla/stream_executor/cuda/version.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/env.h"

namespace stream_executor::gpu {

using GraphNodeHandle = GpuCommandBuffer::GraphNodeHandle;
using GraphConditionalHandle = GpuCommandBuffer::GraphConditionalHandle;
using ConditionType = GpuCommandBuffer::ConditionType;

absl::StatusOr<GpuCommandBuffer::GraphConditionalNodeHandle>
CudaCommandBuffer::CreateConditionalNodeImpl(
    absl::Span<const GraphNodeHandle> dependencies,
    GraphConditionalHandle conditional, ConditionType type) {
  if (stream_exec_->GetDeviceDescription().driver_version() <
      SemanticVersion{12, 3, 0}) {
    return absl::UnimplementedError(
        "Conditional nodes require CUDA driver version >= 12.3");
  }

  // Add conditional node to a graph.
  VLOG(2) << "Add conditional node to a graph " << graph_
          << "; type: " << (type == ConditionType::kIf ? "kIf" : "kWhile")
          << "; conditional: " << conditional << "; deps("
          << dependencies.size()
          << "): " << FormatGraphNodeHandles(dependencies);

  CUgraphNodeParams cu_params{};

  cu_params.type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
  cu_params.conditional.handle = ToCudaGraphHandle(conditional);
  cu_params.conditional.ctx = cuda_context_->context();
  cu_params.conditional.size = 1;

  switch (type) {
    case GpuCommandBuffer::ConditionType::kIf:
      cu_params.conditional.type = CU_GRAPH_COND_TYPE_IF;
      break;
    case GpuCommandBuffer::ConditionType::kWhile:
      cu_params.conditional.type = CU_GRAPH_COND_TYPE_WHILE;
      break;
  }

  std::vector<CUgraphNode> deps = ToCudaGraphHandles(dependencies);
  CUgraphNode node_handle = nullptr;
  RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphAddNode_v2(&node_handle, graph_, deps.data(),
                        /*dependencyData=*/nullptr, deps.size(), &cu_params),
      "Failed to add conditional node to a CUDA graph"));

  auto nested_cmd_buffer = absl::WrapUnique(new CudaCommandBuffer(
      CudaCommandBuffer::Mode::kNested, stream_exec_, cuda_context_,
      cu_params.conditional.phGraph_out[0], /*is_owned_graph=*/false));
  nested_cmd_buffer->parent_ = this;

  return GpuCommandBuffer::GraphConditionalNodeHandle{
      FromCudaGraphHandle(node_handle), std::move(nested_cmd_buffer)};
}

absl::StatusOr<GraphNodeHandle> CudaCommandBuffer::CreateKernelNodeImpl(
    absl::Span<const GraphNodeHandle> dependencies, StreamPriority priority,
    const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& args) {
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();

  VLOG(2) << "Add kernel node to a graph " << graph_
          << "; kernel: " << kernel.name() << "; gdx: " << blocks.x
          << " gdy: " << blocks.y << " gdz: " << blocks.z
          << " bdx: " << threads.x << " bdy: " << threads.y
          << " bdz: " << threads.z << "; shmem: " << shared_mem_bytes
          << "; deps(" << dependencies.size()
          << "): " << FormatGraphNodeHandles(dependencies);

  CUgraphNode node_handle = nullptr;
  const auto& cuda_kernel = static_cast<const CudaKernel&>(kernel);
  CUfunction function = cuda_kernel.gpu_function();
  RETURN_IF_ERROR(
      cuda_kernel.UpdateMaxDynamicSharedMemoryBytes(shared_mem_bytes));

  std::unique_ptr<KernelArgsPackedArrayBase> repacked;
  const KernelArgsPackedArrayBase* packed_args;
  if (cuda_kernel.args_packing()) {
    ASSIGN_OR_RETURN(repacked, cuda_kernel.args_packing()(cuda_kernel, args));
    packed_args = repacked.get();
  } else {
    packed_args = &args;
  }

  auto set_params = [&](auto& params) {
    params.func = function;
    params.gridDimX = blocks.x;
    params.gridDimY = blocks.y;
    params.gridDimZ = blocks.z;
    params.blockDimX = threads.x;
    params.blockDimY = threads.y;
    params.blockDimZ = threads.z;
    params.sharedMemBytes = shared_mem_bytes;
    params.kernelParams =
        // non-const is a requirement of the CUDA API.
        // NOLINTNEXTLINE
        const_cast<void**>(packed_args->argument_addresses().data());
    params.extra = nullptr;
  };

  std::vector<CUgraphNode> deps = ToCudaGraphHandles(dependencies);

  if (stream_exec_->GetDeviceDescription().driver_version() >=
      SemanticVersion{12, 3, 0}) {
    CUgraphNodeParams cu_params{};
    cu_params.type = CU_GRAPH_NODE_TYPE_KERNEL;
    CUDA_KERNEL_NODE_PARAMS_v3& params = cu_params.kernel;
    set_params(params);

    std::vector<CUgraphEdgeData> edge_data;
    edge_data.reserve(deps.size());
    for (size_t i = 0; i < deps.size(); ++i) {
      CUgraphEdgeData edge_data_item{};
      CUgraphNodeType type;
      RETURN_IF_ERROR(cuda::ToStatus(
          cuGraphNodeGetType(deps[i], &type),
          absl::StrCat("Failed to get CUDA graph node type for dependency ",
                       i)));
      if (kernel.use_pdl() && type == CU_GRAPH_NODE_TYPE_KERNEL) {
        edge_data_item.from_port = CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC;
        edge_data_item.type = CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC;
      }
      edge_data.push_back(edge_data_item);
    }
    RETURN_IF_ERROR(cuda::ToStatus(
        cuGraphAddNode_v2(&node_handle, graph_, deps.data(), edge_data.data(),
                          deps.size(), &cu_params),
        "Failed to add kernel node to a CUDA graph"));
  } else {
    if (kernel.use_pdl()) {
      LOG(WARNING)
          << "PDL is not supported for CUDA < 12.3. Falling back to non-PDL.";
    }
    CUDA_KERNEL_NODE_PARAMS params{};
    set_params(params);

    RETURN_IF_ERROR(
        cuda::ToStatus(cuGraphAddKernelNode(&node_handle, graph_, deps.data(),
                                            deps.size(), &params),
                       "Failed to add kernel node to a CUDA graph"));
  }

  if (priority != StreamPriority::Default) {
    CUlaunchAttributeValue value;
    value.priority = stream_exec_->GetGpuStreamPriority(priority);
    RETURN_IF_ERROR(
        cuda::ToStatus(cuGraphKernelNodeSetAttribute(
                           node_handle, CU_LAUNCH_ATTRIBUTE_PRIORITY, &value),
                       "Failed to set kernel node priority"));
  }
  return FromCudaGraphHandle(node_handle);
}

absl::Status CudaCommandBuffer::TraceImpl(
    Stream* stream, absl::AnyInvocable<absl::Status(Stream* stream)> function) {
  if (stream_exec_->GetDeviceDescription().driver_version() <
      SemanticVersion{12, 3, 0}) {
    return absl::UnimplementedError(
        "StreamBeginCaptureToGraph is not implemented for CUDA below version "
        "12.3. Therefore tracing is not supported.");
  }

  RETURN_IF_ERROR(CheckNotFinalized());

  VLOG(5) << "Trace into GPU command buffer graph " << graph_
          << " on a stream: " << stream;

  CudaStream* cuda_stream = static_cast<CudaStream*>(stream);

  uint64_t start_nanos = tsl::Env::Default()->NowNanos();
  {
    ASSIGN_OR_RETURN(
        CudaStream::CaptureHandle capture_handle,
        cuda_stream->BeginCapture(
            graph_, /*dependencies=*/nullptr, /*dependency_data=*/nullptr,
            /*num_dependencies=*/0,
            // THREAD_LOCAL implies that capturing is done only on the current
            // stream. Cuda calls can be made on other streams without
            // interrupting the capture.
            // The default mode CU_STREAM_CAPTURE_MODE_GLOBAL, will capture at
            // at a global level. That would stall everything at a driver level.
            CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    Stream* capture_stream = capture_handle.capturing_stream();
    RETURN_IF_ERROR(function(capture_stream));
    VLOG(5) << "End stream " << capture_stream << " capture";
    RETURN_IF_ERROR(capture_handle.EndCapture());
  }
  uint64_t end_nanos = tsl::Env::Default()->NowNanos();
  VLOG(5) << "Traced into the GPU command buffer graph " << graph_ << " (took "
          << (end_nanos - start_nanos) / 1000 << " μs)";

  size_t num_root_nodes = 0;
  RETURN_IF_ERROR(
      cuda::ToStatus(cuGraphGetRootNodes(graph_, nullptr, &num_root_nodes)));

  if (num_root_nodes == 0) {
    VLOG(5) << "Traced CUDA graph is empty; adding an empty node";
    ASSIGN_OR_RETURN(auto* empty, CreateEmptyCmd({}, StreamPriority::Default));
    (void)empty;
  }

  return absl::OkStatus();
}

absl::StatusOr<GraphConditionalHandle>
CudaCommandBuffer::CreateConditionalHandleImpl() {
  cuda::CUgraphConditionalHandle handle;
  RETURN_IF_ERROR(
      cuda::ToStatus(cuGraphConditionalHandleCreate(
                         &handle, graph_, cuda_context_->context(), 0, 0),
                     "Failed to create conditional handle for a CUDA graph"));
  return FromCudaGraphHandle(handle);
}

}  // namespace stream_executor::gpu

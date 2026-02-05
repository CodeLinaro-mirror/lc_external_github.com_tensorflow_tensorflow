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

#include "xla/tests/codegen_utils.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/Module.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/llvm_compiler.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<std::unique_ptr<Executable>> CompileToExecutable(
    Compiler* compiler, const Compiler::CompileOptions& compile_options,
    std::unique_ptr<HloModule> hlo_module, bool run_optimization_passes) {
  if (run_optimization_passes) {
    TF_ASSIGN_OR_RETURN(hlo_module, compiler->RunHloPasses(
                                        std::move(hlo_module),
                                        /*executor=*/nullptr, compile_options));
  }
  return compiler->RunBackend(std::move(hlo_module), /*executor=*/nullptr,
                              compile_options);
}

namespace {
void IrHook(const llvm::Module& module, std::string& ir) {
  ir = llvm_ir::DumpToString(&module);
}

void SetIrHook(LLVMCompiler* llvm_compiler,
               const LLVMCompiler::ModuleHook& ir_hook,
               bool match_optimized_ir) {
  // Add the IR inspection hook to the LLVM compiler.
  if (match_optimized_ir) {
    llvm_compiler->SetPostOptimizationHook(ir_hook);
  } else {
    llvm_compiler->SetPreOptimizationHook(ir_hook);
  }
}

void ResetIrHook(LLVMCompiler* llvm_compiler) {
  llvm_compiler->RemovePreOptimizationHook();
  llvm_compiler->RemovePostOptimizationHook();
}
}  // namespace

absl::Status CompileAndVerifyIr(LLVMCompiler* compiler,
                                const Compiler::CompileOptions& compile_options,
                                std::unique_ptr<HloModule> hlo_module,
                                const std::string& pattern,
                                bool match_optimized_ir,
                                bool run_optimization_passes) {
  std::string ir;
  SetIrHook(
      compiler, [&ir](const llvm::Module& module) { IrHook(module, ir); },
      match_optimized_ir);
  absl::Status status =
      CompileToExecutable(compiler, compile_options, std::move(hlo_module),
                          run_optimization_passes)
          .status();
  ResetIrHook(compiler);
  TF_RETURN_IF_ERROR(status);

  TF_ASSIGN_OR_RETURN(bool succeeded, RunFileCheck(ir, pattern));
  if (!succeeded) {
    return absl::InternalError(absl::StrCat("FileCheck failed. Full IR: ", ir));
  }
  return absl::OkStatus();
}

}  // namespace xla

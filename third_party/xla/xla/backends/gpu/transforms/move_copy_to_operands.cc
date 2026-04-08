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

#include "xla/backends/gpu/transforms/move_copy_to_operands.h"

#include <algorithm>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using MovedOpsMap =
    absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>;

absl::StatusOr<bool> DoPad(HloInstruction* copy, HloInstruction* pad,
                           HloComputation* computation,
                           std::vector<HloInstruction*>& worklist,
                           MovedOpsMap& moved_ops) {
  HloInstruction* x = pad->mutable_operand(0);
  HloInstruction* c = pad->mutable_operand(1);

  const Layout& target_layout = copy->shape().layout();

  Shape new_copy_shape = x->shape();
  *new_copy_shape.mutable_layout() = target_layout;
  HloInstruction* new_copy = MakeCopyHlo(x, new_copy_shape);

  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_pad,
      MakePadHlo(new_copy, c, pad->padding_config(), &pad->metadata()));
  *new_pad->mutable_shape()->mutable_layout() = target_layout;

  TF_ASSIGN_OR_RETURN(bool changed, computation->ReplaceInstruction(
                                        copy, new_pad,
                                        /*preserve_sharding=*/false));
  if (changed) {
    worklist.push_back(new_copy);
    moved_ops[pad].push_back(new_pad);
  }
  return changed;
}

absl::StatusOr<bool> DoSlice(HloInstruction* copy, HloInstruction* slice,
                             HloComputation* computation,
                             std::vector<HloInstruction*>& worklist,
                             MovedOpsMap& moved_ops) {
  HloInstruction* x = slice->mutable_operand(0);

  const Layout& target_layout = copy->shape().layout();

  Shape new_copy_shape = x->shape();
  *new_copy_shape.mutable_layout() = target_layout;
  HloInstruction* new_copy = MakeCopyHlo(x, new_copy_shape);

  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_slice,
      MakeSliceHlo(new_copy, slice->slice_starts(), slice->slice_limits(),
                   slice->slice_strides(), &slice->metadata()));
  *new_slice->mutable_shape()->mutable_layout() = target_layout;

  TF_ASSIGN_OR_RETURN(bool changed, computation->ReplaceInstruction(
                                        copy, new_slice,
                                        /*preserve_sharding=*/false));
  if (changed) {
    worklist.push_back(new_copy);
    moved_ops[slice].push_back(new_slice);
  }
  return changed;
}

absl::StatusOr<bool> DoDynamicSlice(HloInstruction* copy, HloInstruction* ds,
                                    HloComputation* computation,
                                    std::vector<HloInstruction*>& worklist,
                                    MovedOpsMap& moved_ops) {
  HloInstruction* x = ds->mutable_operand(0);

  const Layout& target_layout = copy->shape().layout();

  Shape new_copy_shape = x->shape();
  *new_copy_shape.mutable_layout() = target_layout;
  HloInstruction* new_copy = MakeCopyHlo(x, new_copy_shape);

  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_ds,
      MakeDynamicSliceHlo(
          new_copy,
          absl::Span<HloInstruction* const>(ds->operands()).subspan(1),
          ds->dynamic_slice_sizes(), &ds->metadata()));
  *new_ds->mutable_shape()->mutable_layout() = target_layout;

  TF_ASSIGN_OR_RETURN(bool changed, computation->ReplaceInstruction(
                                        copy, new_ds,
                                        /*preserve_sharding=*/false));
  if (changed) {
    worklist.push_back(new_copy);
    moved_ops[ds].push_back(new_ds);
  }
  return changed;
}

absl::StatusOr<bool> DoReduceWindow(HloInstruction* copy, HloInstruction* rw,
                                    HloComputation* computation,
                                    std::vector<HloInstruction*>& worklist,
                                    MovedOpsMap& moved_ops) {
  if (rw->shape().IsTuple()) {
    return false;
  }
  HloInstruction* x = rw->mutable_operand(0);
  HloInstruction* init = rw->mutable_operand(1);

  const Layout& target_layout = copy->shape().layout();

  Shape new_copy_shape = x->shape();
  *new_copy_shape.mutable_layout() = target_layout;
  HloInstruction* new_copy = MakeCopyHlo(x, new_copy_shape);

  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_rw,
      MakeReduceWindowHlo(new_copy, init, rw->window(),
                          rw->called_computations()[0], &rw->metadata()));
  *new_rw->mutable_shape()->mutable_layout() = target_layout;

  TF_ASSIGN_OR_RETURN(bool changed, computation->ReplaceInstruction(
                                        copy, new_rw,
                                        /*preserve_sharding=*/false));
  if (changed) {
    worklist.push_back(new_copy);
    moved_ops[rw].push_back(new_rw);
  }
  return changed;
}

absl::StatusOr<bool> DoReverse(HloInstruction* copy, HloInstruction* reverse,
                               HloComputation* computation,
                               std::vector<HloInstruction*>& worklist,
                               MovedOpsMap& moved_ops) {
  HloInstruction* x = reverse->mutable_operand(0);

  const Layout& target_layout = copy->shape().layout();

  Shape new_copy_shape = x->shape();
  *new_copy_shape.mutable_layout() = target_layout;
  HloInstruction* new_copy = MakeCopyHlo(x, new_copy_shape);

  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_reverse,
      MakeReverseHlo(new_copy, reverse->dimensions(), &reverse->metadata()));
  *new_reverse->mutable_shape()->mutable_layout() = target_layout;

  TF_ASSIGN_OR_RETURN(bool changed, computation->ReplaceInstruction(
                                        copy, new_reverse,
                                        /*preserve_sharding=*/false));
  if (changed) {
    worklist.push_back(new_copy);
    moved_ops[reverse].push_back(new_reverse);
  }
  return changed;
}

absl::StatusOr<bool> DoConvert(HloInstruction* copy, HloInstruction* convert,
                               HloComputation* computation,
                               std::vector<HloInstruction*>& worklist,
                               MovedOpsMap& moved_ops) {
  HloInstruction* x = convert->mutable_operand(0);

  const Layout& target_layout = copy->shape().layout();

  Shape new_copy_shape = x->shape();
  *new_copy_shape.mutable_layout() = target_layout;
  HloInstruction* new_copy = MakeCopyHlo(x, new_copy_shape);

  HloInstruction* new_convert = MakeConvertToHlo(
      new_copy, convert->shape().element_type(), &convert->metadata());
  *new_convert->mutable_shape()->mutable_layout() = target_layout;

  TF_ASSIGN_OR_RETURN(bool changed, computation->ReplaceInstruction(
                                        copy, new_convert,
                                        /*preserve_sharding=*/false));
  if (changed) {
    worklist.push_back(new_copy);
    moved_ops[convert].push_back(new_convert);
  }
  return changed;
}

absl::StatusOr<bool> DoElementwise(HloInstruction* copy, HloInstruction* hlo,
                                   HloComputation* computation,
                                   std::vector<HloInstruction*>& worklist,
                                   MovedOpsMap& moved_ops) {
  const Layout& target_layout = copy->shape().layout();

  std::vector<HloInstruction*> new_operands;
  new_operands.reserve(hlo->operand_count());
  for (int64_t i = 0; i < hlo->operand_count(); ++i) {
    HloInstruction* x = hlo->mutable_operand(i);

    Shape new_copy_shape = x->shape();
    *new_copy_shape.mutable_layout() = target_layout;
    HloInstruction* new_copy = MakeCopyHlo(x, new_copy_shape);
    new_operands.push_back(new_copy);
    worklist.push_back(new_copy);
  }

  std::unique_ptr<HloInstruction> new_hlo =
      hlo->CloneWithNewOperands(copy->shape(), new_operands);

  HloInstruction* new_hlo_ptr = new_hlo.get();
  TF_RETURN_IF_ERROR(
      computation->ReplaceWithNewInstruction(copy, std::move(new_hlo)));

  moved_ops[hlo].push_back(new_hlo_ptr);

  return true;
}

absl::StatusOr<bool> DoConcatenate(HloInstruction* copy, HloInstruction* concat,
                                   HloComputation* computation,
                                   std::vector<HloInstruction*>& worklist,
                                   MovedOpsMap& moved_ops) {
  const HloInstruction* first = concat->operand(0);
  const Layout& first_layout = first->shape().layout();

  for (const HloInstruction* op : concat->operands()) {
    if (op->shape().layout() != first_layout) {
      return false;
    }
  }

  const Layout& target_layout = copy->shape().layout();

  std::vector<HloInstruction*> new_operands;
  new_operands.reserve(concat->operand_count());
  for (HloInstruction* op : concat->mutable_operands()) {
    Shape new_copy_shape = op->shape();
    *new_copy_shape.mutable_layout() = target_layout;
    new_operands.push_back(MakeCopyHlo(op, new_copy_shape));
  }

  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_concat,
      MakeConcatHlo(new_operands, concat->concatenate_dimension()));
  *new_concat->mutable_shape()->mutable_layout() = target_layout;

  TF_ASSIGN_OR_RETURN(bool changed, computation->ReplaceInstruction(
                                        copy, new_concat,
                                        /*preserve_sharding=*/false));
  if (changed) {
    for (HloInstruction* op : new_operands) {
      worklist.push_back(op);
    }
    moved_ops[concat].push_back(new_concat);
  }
  return changed;
}

}  // end namespace

absl::StatusOr<bool> MoveCopyToOperands::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> post_order =
        computation->MakeInstructionPostOrder();

    std::vector<HloInstruction*> worklist = post_order;
    MovedOpsMap moved_ops;

    while (!worklist.empty()) {
      HloInstruction* hlo = worklist.back();
      worklist.pop_back();

      if (computation->IsMarkedAsDead(hlo)) {
        continue;
      }

      if (hlo->opcode() != HloOpcode::kCopy) {
        continue;
      }

      HloInstruction* operand = hlo->mutable_operand(0);
      if (hlo->shape().layout() == operand->shape().layout()) {
        TF_ASSIGN_OR_RETURN(bool local_changed,
                            computation->ReplaceInstruction(
                                hlo, operand, /*preserve_sharding=*/false));
        changed |= local_changed;
        continue;
      }

      // Try to reuse an existing moved op.
      const Layout& target_layout = hlo->shape().layout();
      auto it = moved_ops.find(operand);
      if (it != moved_ops.end()) {
        bool reused = false;
        for (HloInstruction* new_op : it->second) {
          if (new_op->shape().layout() == target_layout) {
            TF_RETURN_IF_ERROR(computation->ReplaceInstruction(hlo, new_op));
            reused = true;
            changed = true;
            break;
          }
        }
        if (reused) {
          continue;
        }
      }

      // Parameter folding logic
      if (operand->opcode() == HloOpcode::kParameter) {
        if (computation == module->entry_computation() &&
            module->config()
                .debug_options()
                .xla_pjrt_allow_auto_layout_in_hlo()) {
          bool all_users_agree = true;
          for (const HloInstruction* user : operand->users()) {
            if (user->opcode() != HloOpcode::kCopy ||
                user->shape().layout() != hlo->shape().layout()) {
              all_users_agree = false;
              break;
            }
          }
          if (all_users_agree) {
            *operand->mutable_shape()->mutable_layout() = hlo->shape().layout();
            TF_RETURN_IF_ERROR(
                module->mutable_entry_computation_layout()
                    ->mutable_parameter_layout(operand->parameter_number())
                    ->CopyLayoutFromShape(hlo->shape()));
            TF_ASSIGN_OR_RETURN(bool local_changed,
                                computation->ReplaceInstruction(
                                    hlo, operand, /*preserve_sharding=*/false));
            changed |= local_changed;
            continue;
          }
        }
      }

      absl::StatusOr<bool> local_changed = false;
      switch (operand->opcode()) {
        case HloOpcode::kPad:
          local_changed = DoPad(hlo, operand, computation, worklist, moved_ops);
          break;
        case HloOpcode::kSlice:
          local_changed =
              DoSlice(hlo, operand, computation, worklist, moved_ops);
          break;
        case HloOpcode::kDynamicSlice:
          local_changed =
              DoDynamicSlice(hlo, operand, computation, worklist, moved_ops);
          break;
        case HloOpcode::kReduceWindow:
          local_changed =
              DoReduceWindow(hlo, operand, computation, worklist, moved_ops);
          break;
        case HloOpcode::kReverse:
          local_changed =
              DoReverse(hlo, operand, computation, worklist, moved_ops);
          break;
        case HloOpcode::kConvert:
          local_changed =
              DoConvert(hlo, operand, computation, worklist, moved_ops);
          break;
        case HloOpcode::kConcatenate:
          local_changed =
              DoConcatenate(hlo, operand, computation, worklist, moved_ops);
          break;
        default:
          if (operand->IsElementwise()) {
            local_changed =
                DoElementwise(hlo, operand, computation, worklist, moved_ops);
          }
          break;
      }

      TF_RETURN_IF_ERROR(local_changed.status());
      changed |= local_changed.value();
    }
  }

  return changed;
}

}  // namespace xla::gpu

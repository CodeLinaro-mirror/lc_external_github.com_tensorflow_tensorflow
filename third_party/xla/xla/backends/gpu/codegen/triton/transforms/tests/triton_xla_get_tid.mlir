// RUN: xla-opt %s --triton-xla-get-tid | FileCheck %s

// CHECK-LABEL: tt.func @kernel()
tt.func @kernel() -> i32 {
  // CHECK-NOT: triton_xla.get_flat_tid

  // Check for the inline assembly to get thread IDs (tid.x, tid.y, tid.z)
  // CHECK: %[[TID:.*]] = tt.elementwise_inline_asm
  // CHECK-SAME: mov.u32 $0, %tid.x;

  // CHECK: tt.return %[[TID]] : i32
  %0 = triton_xla.get_tid : () -> i32
  tt.return %0 : i32
}
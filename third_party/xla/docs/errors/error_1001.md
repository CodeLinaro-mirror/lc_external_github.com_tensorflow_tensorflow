# Error code: 1001

**Category:** Compile Time Vmem OOM

This error indicates that the program requires more Scoped Vector Memory (Vmem)
than is allocated or physically available on the TPU device.

**Sample Error Messages:**

```
RESOURCE_EXHAUSTED: Ran out of memory in memory space vmem while allocating on stack for %ragged_latency_optimized_all_gather_lhs_contracting_gated_matmul_kernel.18 = bf16[2048,4096]{1,0:T(8,128)(2,1)} custom-call(...) ...
```

**XLA Backends:** TPU

## Overview

TPUs have Vector Memory (VMEM) which is a local scratchpad memory used
exclusively by the TensorCore (TC). The compiler manages Vmem for several
types of allocations:

* **Scoped Allocations:** Storage for temporaries and intermediate results
during an HLO execution.
* **Constrained Temporaries:** HLO temporaries explicitly constrained to Vmem by
a custom kernel (e.g., using [pallas.tpu.with_memory_space_constraint](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.with_memory_space_constraint.html)).
* **Register Spills:** Overflow when registers are insufficient for active
computations.

A Compile time Vmem OOM typically occurs either due to an internal
compiler bug or when a specific custom kernel requests more Scoped Vmem to
execute than has been allocated to it.

## Debugging

Carefully analyze the error message to identify if the error stems from a
custom kernel or a standard HLO.

* **Custom Kernel Vmem OOM**:
If the error or the largest allocations point to a custom kernel → Jump to
[Retune the Kernel](#retune-kernel).
* **Non-Kernel Vmem Issues**:
If the Vmem OOM occurs due to a non-custom-kernel op, this is likely an internal
compiler bug. Please file a bug on XLA with an HLO dump.

---

### Retune the Kernel {#retune-kernel}

If the error originates from a custom kernel, use the following techniques to
reduce memory pressure:

* **Adjust Block Sizes:** Reduce the block sizes (tile sizes) in your kernel
configuration to lower Vmem usage.
* **Set Per-Kernel Vmem Limits:** Explicitly request the required amount of
memory for that specific kernel using the [vmem_limit_bytes param](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.CompilerParams.html#jax.experimental.pallas.tpu.CompilerParams.vmem_limit_bytes)
* **Modify Memory Coloring:** If inputs/outputs are explicitly colored to VMEM
using [pallas.tpu.with_memory_space_constraint](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.with_memory_space_constraint.html)
try relaxing these constraints to free up space. Ensure the combination of
colored operands and stack usage does not exceed the limit.
* If kernel specific retuning is difficult or the issue affects many kernels,
you can adjust the global Vmem limit using the flag
[--xla_tpu_scoped_vmem_limit_kib](https://openxla.org/xla/flags_guidance).
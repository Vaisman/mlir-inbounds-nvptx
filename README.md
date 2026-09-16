# NVIDIA datapoint: `in_bounds` vs masking on the NVPTX path

Companion reproducers for the Discourse RFC *"Should `vector.transfer_read`/`write` keep
`in_bounds`? Measurements on the masking alternative"*, which grew out of llvm-project PR #215340.

The thread contains CPU datapoints for AArch64 and x86. This package adds the NVIDIA side:
MLIR to LLVM IR to the NVPTX backend to PTX.

## How to run

```bash
LLVM_SOURCE_ROOT=/path/to/llvm-project BIN=/path/to/llvm-project/build/bin ./run.sh
```

Needs `mlir-opt`, `mlir-translate` and `llc` built with `NVPTX` in `LLVM_TARGETS_TO_BUILD`.
Both paths are required: `LLVM_SOURCE_ROOT` identifies the source checkout for provenance, while
`BIN` points to the built tools (which may live in an out-of-tree build).
No GPU and no CUDA toolkit required: everything here is static PTX emitted by `llc`.

## A. Removing `in_bounds` blocks the current Tensor Core path

`A1_tensorcore_in_bounds.mlir`, `A2_tensorcore_masked.mlir`, and
`A3_tensorcore_no_in_bounds.mlir` are the same 16x8x16 f16 GEMM tile feeding a
`vector.contract`. They use **dynamic**
`memref<?x?xf16, #gpu.address_space<workgroup>>`, so the folder cannot derive boundedness from the
types.

```
A1_tensorcore_in_bounds          ldmatrix=3 mma.sync=1 transfer_read_left=0 contract_left=0
A2_tensorcore_masked             ldmatrix=0 mma.sync=0 transfer_read_left=3 contract_left=1
A3_tensorcore_no_in_bounds       ldmatrix=0 mma.sync=0 transfer_read_left=3 contract_left=1
```

With `in_bounds = [true, true]` the tile becomes three `nvgpu.ldmatrix` and one `nvgpu.mma.sync`.
The explicit-runtime-mask arm has no `in_bounds` attribute, so it does not use the
mask-plus-`in_bounds` combination marked for future restriction in `VectorOps.td`. Its transfers
and contract remain. `A3_tensorcore_no_in_bounds.mlir` is a mask-free diagnostic control, not the
proposal itself: it shows the form left if A2's mask can be proved all-true and eliminated. The
dynamic memref extents still provide no proof that the tile is in bounds, so it reaches the same
conversion cliff.

`A1_tensorcore_kernel.mlir` also carries the positive arm through
`mlir-opt -> mlir-translate -> llc`; its PTX contains three `ldmatrix.sync` and one
`mma.sync`.

This is not a cost difference, it is a capability cliff, and it is structural. Every relevant
warp-level matrix lowering requires both no mask and no out-of-bounds dimension:

| location | gate |
|---|---|
| `mlir/lib/Conversion/VectorToGPU/VectorToGPU.cpp:174` | `transfer_read` to `gpu.subgroup_mma` |
| `mlir/lib/Conversion/VectorToGPU/VectorToGPU.cpp:200` | `transfer_write` to `gpu.subgroup_mma` |
| `mlir/lib/Conversion/VectorToGPU/VectorToGPU.cpp:506` | `nvgpu.mma.sync` path |
| `mlir/lib/Dialect/NVGPU/Utils/MMAUtils.cpp:270,297` | `canLowerToWarpMatrixOperation` |

All of them read `op.getMask() || op.hasOutOfBoundsDim()` and bail.

## B, C. What a masked load costs in PTX

`C_transfer_matrix.mlir`, a 4xf32 read/write over `memref<?xf32>`:

| function | PTX instructions | branches | loads |
|---|---|---|---|
| `read_in_bounds` | 10 | 0 | 4 |
| `read_maybe_oob` (no `in_bounds`, no mask) | 28 | 4 | 4 |
| `read_explicit_mask_in_bounds` | 27 | 4 | 4 |
| `read_explicit_mask_maybe_oob` | 35 | 4 | 4 |
| `read_all_true_mask_in_bounds` (`constant_mask [4]`) | 10 | 0 | 4 |
| `read_partial_constant_mask_in_bounds` (`constant_mask [3]`) | 10 | 0 | 3 |
| `write_in_bounds` | 10 | 0 | 0 |
| `write_maybe_oob` | 22 | 4 | 0 |

The rows combining a mask with `in_bounds` are deliberate controls for the current IR: they
separate mask cost from bounds checks. They are not proposed as the replacement form.

`B*.mlir` is the same experiment at `vector<8xf32>`: 15 instructions and 0 branches with
`in_bounds`, against 49 instructions and 8 branches with a runtime mask.

The instruction and branch counts for the complete 4xf32 matrix are identical with `sm_70`,
`sm_80`, and `sm_90`.

For these NVPTX cases, the important backend dividing line is whether the mask is known at compile
time:

- Both the all-true `constant_mask [4]` and the *partial* `constant_mask [3]` produce
  straight-line code with no branches. For the partial one the backend emits three loads and fills
  the inactive lane from the padding value. Thus this is not only an all-true-mask effect.
- A runtime mask costs one `setp` plus `@%p bra` per lane. NVPTX TTI rejects runtime masked loads,
  so `llvm.masked.load` is scalarized into per-lane conditional blocks. In this reproducer the
  boundary input is uniform, so the branches do not imply warp divergence. In a typical tiled GEMM
  a boundary derived from a block or thread index can vary across threads and then diverge.
- Dropping `in_bounds` is itself enough to trigger this. `read_maybe_oob` has no explicit mask at
  all and still lands on the branchy path, because on a dynamic `memref` nothing can prove the
  access safe.

More precisely, the NVPTX TTI supports masked accesses only when
`MaskKind::ConstantMask`. Runtime masks are rejected and fall back to generic scalarization.
`isLegalMaskedStore` is narrower still: it requires at least 32-byte alignment, target support
for 256-bit accesses, and exactly eight 32-bit or four 64-bit elements. These restrictions were
added by commit `17852deda7fb` (`[NVPTX] Lower LLVM masked vector loads and stores to PTX`,
2025-11-25). They explain why `write_maybe_oob` also takes the branchy path.

In the kernel-ABI control in section D, neither arm becomes `ld.global.v4.b32`: both use four
scalar `ld.global.b32` because alignment of the descriptor-provided base pointer cannot be proved.
The instruction-count difference there therefore isolates control-flow overhead rather than a
change in vector load width.

## D. Same thing with a real kernel ABI

`D_kernel_abi.mlir` repeats the comparison with a real `gpu.func ... kernel` entry point and
global address space. Unlike A/B/C, it has static memref shapes, but the transfer index is
dynamic:

| kernel | PTX instructions | branches | `ld.global` |
|---|---|---|---|
| `kernel_in_bounds` | 14 | 0 | 4 |
| `kernel_maybe_oob` | 31 | 4 | 4 |

## What this implies for the RFC

Section A is the load-bearing result. With today's GPU conversion, either an explicit mask or an
unproved out-of-bounds dimension prevents the transfer from reaching Tensor Core operations.
Removing `in_bounds` therefore requires its boundedness guarantee to be carried in another IR
form, or rederived where the IR contains sufficient size and index constraints, before
`convert-vector-to-gpu`; eliminating an explicit all-true mask alone does not cover the mask-free
A3 case.

The mechanism meant for that job is `vector::eliminateVectorMasks`, and today it cannot do it:

- it is scalable-only, `mlir/lib/Dialect/Vector/Transforms/VectorMaskElimination.cpp:99` returns
  immediately when there is no `vscaleRange`;
- its only in-tree caller is `mlir/test/lib/Dialect/Vector/TestVectorTransforms.cpp`, so it runs in
  no pipeline at all.

PR #221595 proposes a fix for the first, but it is still open, so fixed-size mask elimination is
not in `main`; the second is unaddressed. This is one part of the GPU problem, while A3 also shows
that the no-mask representation needs a way to communicate or recover the same proof.

Separately, a masked transfer cannot be unrolled today:
`mlir/lib/Dialect/Vector/Transforms/VectorUnroll.cpp:162` and `:217` bail out when the transfer
carries a mask. Unrolling to a hardware vector shape is how a tile reaches an mma instruction, so
this compounds section A.

## Provenance and limits

- Checkout llvm-project `f0d41abb33b6`.
- `mlir-opt`, `mlir-translate`, and `llc` were all rebuilt from that checkout on 2026-09-15 before
  the recorded run. `run.sh` prints the HEAD of `LLVM_SOURCE_ROOT` and all three binary timestamps.
- The working tree contained two unrelated uncommitted changes in NVGPU shared-memory
  optimization. None of the four pipelines in `run.sh` invokes `nvgpu-optimize-shared-memory`.
- `llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80`.
- Counts are static PTX, produced by `run.sh`. They include prologue and epilogue, so read the
  ratios rather than the absolute numbers.
- No NVIDIA GPU was available here, so there is no `ptxas`, no SASS and no runtime figure. A
  wall-clock number comparable to the PolyBench result in the thread still needs hardware.

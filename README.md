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

This is not a cost difference, it is a capability cliff, and it is structural. The relevant GPU
conversion checks below all require an unmasked, in-bounds transfer:

| location | guarded path |
|---|---|
| `mlir/lib/Conversion/VectorToGPU/VectorToGPU.cpp:174` | `transferReadSupportsMMAMatrixType` (`gpu.subgroup_mma` read) |
| `mlir/lib/Conversion/VectorToGPU/VectorToGPU.cpp:200` | `transferWriteSupportsMMAMatrixType` (`gpu.subgroup_mma` write) |
| `mlir/lib/Conversion/VectorToGPU/VectorToGPU.cpp:506` | `CombineTransferReadOpTranspose` (fold transpose into transfer read) |
| `mlir/lib/Dialect/NVGPU/Utils/MMAUtils.cpp:270,297` | `nvgpu::canLowerToWarpMatrixOperation` (read/write on the `nvgpu.mma.sync` path) |

Each has an equivalent `getMask() || hasOutOfBoundsDim()` guard.

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

Separately, `vector::eliminateVectorMasks` can replace a provably all-true `vector.create_mask`
with a constant mask, but it cannot supply the missing boundedness guarantee in A3. At the time of
the original experiment on `f0d41abb33b6`, even that mask elimination was unavailable for
fixed-size vectors in production pipelines:

- it was scalable-only: `mlir/lib/Dialect/Vector/Transforms/VectorMaskElimination.cpp:99`
  returned immediately when there was no `vscaleRange` (#221595 was still open);
- its only in-tree caller is `mlir/test/lib/Dialect/Vector/TestVectorTransforms.cpp`, so it runs in
  no pipeline at all.

#221595 has since landed (it is an ancestor of `fc99ddb16876`), which removes the first
limitation; the second still holds. Section H below shows what fixed-size elimination does and
does not remove. This is one part of the GPU problem, while A3 also shows that the no-mask
representation needs a way to communicate or recover the same proof.

Separately, `VectorUnroll` handles the two mask representations inconsistently today. A transfer
with a mask operand is left unchanged (`mlir/lib/Dialect/Vector/Transforms/VectorUnroll.cpp:162`
and `:217` bail out), while an operation inside `vector.mask` is unrolled in place and produces
invalid IR, since the region then holds several ops. Unrolling to a hardware vector
shape is how a tile reaches an mma instruction, so this compounds section A.

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

## H, I. Masks that stay: peeled remainder tiles

`eliminateVectorMasks` removes masks that are provably all-true. Peeling produces tiles where the
masks are genuinely partial, so they stay, and the GPU lowering has to cope with `vector.mask`
next to unmasked tiles in the same function.

- `H_peeled_matmul.mlir`: a dynamic `linalg.matmul` (`memref<?x?xf16>`) tiled to `[16, 8, 16]`,
  peeled in all three loops and vectorized with `vector_sizes [16, 8, 16] create_named_contraction`.
  Lowered with `-convert-vector-to-gpu="use-nvgpu=true"`.
- `I_peeled_matmul_wmma.mlir`: the vendor-neutral `gpu.subgroup_mma` path. Static row strides,
  dynamic `M`, `B` stored as `(n, k)`; only the `M` loop is peeled. Lowered with
  `-convert-vector-to-gpu`.

Two builds are compared; the script prints the md5 of the `mlir-opt` it runs:

```bash
# llvm-project fc99ddb16876 (includes #221595); mlir-opt md5 c34b9d02c214d872d979462714a5c8d8
BIN=/path/to/baseline/build/bin ./run_masked_remainders.sh
# fc99ddb16876 + 848cc47c636e (#226732); mlir-opt md5 f2fd969ff25a6b32d19e1d7b80fee8d9
BIN=/path/to/patched/build/bin ./run_masked_remainders.sh
```

The masks are the same with both builds. The vectorizer masks every contraction;
`-canonicalize` folds the masks of the main loop, whose tiles are static; the remainder loops keep
theirs, and `eliminateVectorMasks` removes nothing further:

```
== H_peeled_matmul
after vectorize:            contract=8 masked_contract=8 vector.mask=40
after canonicalize:         contract=8 masked_contract=7 vector.mask=31
after eliminateVectorMasks: contract=8 masked_contract=7 vector.mask=31
== I_peeled_matmul_wmma
after vectorize:            contract=2 masked_contract=2 vector.mask=8
after canonicalize:         contract=2 masked_contract=1 vector.mask=4
after eliminateVectorMasks: contract=2 masked_contract=1 vector.mask=4
```

Baseline: preparing the first masked remainder makes the pass fail, so no usable result is
produced, including for the unmasked main loop:

```
H_nvgpu            exit=1 error: 'vector.mask' op expects only one operation to mask
I_subgroup_mma     exit=1 error: 'vector.mask' op expects only one operation to mask
```

The MMA preparation patterns (`CanonicalizeContractMatmulToMMT` on the nvgpu path,
`PrepareContractToGPUMMA` on the `gpu.subgroup_mma` path) rewrite a remainder contraction in
place and leave extra ops inside its `vector.mask` region: a `vector.transpose` in H, and in I the
same transpose already folded into a new `vector.transfer_read`.

Patched (#226732): masked contractions are left alone by the preparation patterns, the masked
remainders stay unchanged, and the eligible main loop lowers to one MMA operation:

```
H_nvgpu            exit=0 mma.sync=1 subgroup_mma_compute=0 vector.mask=31
I_subgroup_mma     exit=0 mma.sync=0 subgroup_mma_compute=1 vector.mask=4
```

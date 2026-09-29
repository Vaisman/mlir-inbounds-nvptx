# Masked Vector operations in GPU pipelines

Companion reproducers for the Discourse RFC
*[Should `vector.transfer_read`/`write` keep `in_bounds`? Measurements on the masking
alternative](https://discourse.llvm.org/t/91649)*.

The primary examples, H and I, start from an upstream `linalg.matmul` pipeline and show why
genuinely partial masks remain on peeled tiles. The earlier A-D examples record the current
NVPTX lowering and its generated PTX.

## How to run

For the peeled-remainder examples H and I:

```bash
BIN=/path/to/llvm-project/build/bin ./run_masked_remainders.sh
```

This requires `mlir-opt` with the MLIR test passes enabled. For the earlier NVPTX examples A-D:

```bash
LLVM_SOURCE_ROOT=/path/to/llvm-project BIN=/path/to/llvm-project/build/bin ./run.sh
```

The A-D script needs `mlir-opt`, `mlir-translate` and `llc` built with `NVPTX` in
`LLVM_TARGETS_TO_BUILD`. `LLVM_SOURCE_ROOT` identifies the source checkout for provenance, while
`BIN` points to the built tools. No GPU or CUDA toolkit is required: the script emits static PTX.

## H, I. Masks that stay: peeled remainder tiles

`eliminateVectorMasks` removes masks that are provably all-true. Peeling produces tiles whose
masks are genuinely partial, so they stay, and GPU conversion has to coexist with masked and
unmasked tiles in the same function.

- `H_peeled_matmul.mlir`: a dynamic `linalg.matmul` (`memref<?x?xf16>`) tiled to `[16, 8, 16]`,
  peeled in all three loops and vectorized with `vector_sizes [16, 8, 16] create_named_contraction`.
  Lowered with `-convert-vector-to-gpu="use-nvgpu=true"`.
- `I_peeled_matmul_wmma.mlir`: the vendor-neutral `gpu.subgroup_mma` path. Static row strides,
  dynamic `M`, `B` stored as `(n, k)`; only the `M` loop is peeled. Lowered with
  `-convert-vector-to-gpu`.

Two builds are compared; the script prints the md5 of the `mlir-opt` it runs:

```bash
# llvm-project fc99ddb16876 (includes llvm-project#221595)
# mlir-opt md5 c34b9d02c214d872d979462714a5c8d8
BIN=/path/to/baseline/build/bin ./run_masked_remainders.sh

# fc99ddb16876 + 0f87d4ee7e91 (tree-identical to llvm-project#226732 head 848cc47c636e)
# mlir-opt md5 f2fd969ff25a6b32d19e1d7b80fee8d9
BIN=/path/to/patched/build/bin ./run_masked_remainders.sh
```

The vectorizer masks every contraction. `-canonicalize` folds the masks of the full main tile,
whose sizes are static. The remainder loops keep their masks, and `eliminateVectorMasks` removes
nothing further:

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

On the baseline, preparing the first masked remainder makes the pass fail, so it produces no
usable result for the function, including the unmasked main loop:

```
H_nvgpu            exit=1 error: 'vector.mask' op expects only one operation to mask
I_subgroup_mma     exit=1 error: 'vector.mask' op expects only one operation to mask
```

The MMA preparation patterns (`CanonicalizeContractMatmulToMMT` on the nvgpu path and
`PrepareContractToGPUMMA` on the `gpu.subgroup_mma` path) rewrite a remainder contraction in
place and leave extra operations inside its `vector.mask` region: a `vector.transpose` in H, and
in I the same transpose already folded into a new `vector.transfer_read`.

With [llvm-project#226732](https://github.com/llvm/llvm-project/pull/226732), masked contractions
are left unchanged by the preparation patterns. The pass succeeds, the eligible main loop lowers
to one MMA operation, and the masked remainders remain in Vector IR:

```
H_nvgpu            exit=0 mma.sync=1 subgroup_mma_compute=0 vector.mask=31
I_subgroup_mma     exit=0 mma.sync=0 subgroup_mma_compute=1 vector.mask=4
```

## A. Current GPU conversion distinguishes `in_bounds`, masks and neither

`A1_tensorcore_in_bounds.mlir`, `A2_tensorcore_masked.mlir`, and
`A3_tensorcore_no_in_bounds.mlir` are the same 16x8x16 f16 GEMM tile feeding a
`vector.contract`. They use dynamic `memref<?x?xf16, #gpu.address_space<workgroup>>`. These files
record how the current IR and GPU conversion distinguish an `in_bounds` transfer, a runtime-masked
transfer, and a transfer with neither property.

```
A1_tensorcore_in_bounds          ldmatrix=3 mma.sync=1 transfer_read_left=0 contract_left=0
A2_tensorcore_masked             ldmatrix=0 mma.sync=0 transfer_read_left=3 contract_left=1
A3_tensorcore_no_in_bounds       ldmatrix=0 mma.sync=0 transfer_read_left=3 contract_left=1
```

With `in_bounds = [true, true]`, A1 becomes three `nvgpu.ldmatrix` operations and one
`nvgpu.mma.sync`. A2 has a runtime mask and remains in Vector IR. A3 is a diagnostic control for
today's semantics: because it has neither a mask nor `in_bounds`, current conversion also leaves it
in Vector IR.

A3 is **not** a blocker for the proposed mask-only semantics. Under the proposed rule, an
out-of-bounds access must be masked and is otherwise undefined, so an unmasked transfer is in
bounds by definition. Once `in_bounds` is removed, GPU conversion can treat the A3 form as
eligible without recovering a separate bounds proof. A2 remains the relevant case: a genuinely
partial mask cannot be eliminated and must either be handled or cause a safe fallback.

`A1_tensorcore_kernel.mlir` also carries the positive arm through
`mlir-opt -> mlir-translate -> llc`; its PTX contains three `ldmatrix.sync` and one
`mma.sync`.

The current results come from these GPU conversion checks:

| location | guarded path |
|---|---|
| `mlir/lib/Conversion/VectorToGPU/VectorToGPU.cpp:174` | `transferReadSupportsMMAMatrixType` (`gpu.subgroup_mma` read) |
| `mlir/lib/Conversion/VectorToGPU/VectorToGPU.cpp:200` | `transferWriteSupportsMMAMatrixType` (`gpu.subgroup_mma` write) |
| `mlir/lib/Conversion/VectorToGPU/VectorToGPU.cpp:506` | `CombineTransferReadOpTranspose` (fold transpose into transfer read) |
| `mlir/lib/Dialect/NVGPU/Utils/MMAUtils.cpp:270,297` | `nvgpu::canLowerToWarpMatrixOperation` (read/write on the `nvgpu.mma.sync` path) |

Each currently has an equivalent `getMask() || hasOutOfBoundsDim()` guard. The mask half remains
relevant for partial tiles. The `hasOutOfBoundsDim()` half reflects the current `in_bounds`
semantics and is not evidence that the proposed no-mask form needs another boundedness marker.

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
- Under today's semantics, omitting `in_bounds` is enough to trigger this. `read_maybe_oob` has no
  explicit mask and still lands on the branchy path. This is a measurement of the current
  lowering, not the proposed semantics: if an unmasked out-of-bounds access becomes undefined,
  the no-mask form can be lowered as in-bounds instead.

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

H and I are the main result: an upstream tiling, peeling and vectorization pipeline naturally
produces a mix of unmasked main tiles and genuinely masked remainder tiles. The main-tile masks
fold away, while the remainder masks cannot. Vector and GPU patterns therefore have to preserve
region masks or bail out safely; otherwise one remainder can invalidate the whole function and
prevent conversion of the eligible main tile.

[llvm-project#221595](https://github.com/llvm/llvm-project/pull/221595) added fixed-size support to
`eliminateVectorMasks`. H and I include that change. It does not alter their remainder masks,
because those masks are partial rather than missed all-true cases.

A-D answer a different question: what the current NVPTX path emits when a runtime mask, or the
absence of `in_bounds`, reaches lower-level conversion. The runtime-mask cost remains useful data,
but A3 is only a control for today's semantics. It does not show that removing `in_bounds` requires
another boundedness representation if unmasked out-of-bounds accesses become undefined.

Separately, `VectorUnroll` handles the two mask representations inconsistently today. A transfer
with a mask operand is left unchanged (`mlir/lib/Dialect/Vector/Transforms/VectorUnroll.cpp:162`
and `:217` bail out), while an operation inside `vector.mask` is unrolled in place and produces
invalid IR, since the region then holds several ops. Unrolling to a hardware vector
shape is how a tile reaches an MMA instruction, so it is another place that must preserve a mask
or bail out before the migration.

## Provenance and limits for A-D

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

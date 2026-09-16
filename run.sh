#!/usr/bin/env bash
# NVIDIA/NVPTX datapoint for the `in_bounds` vs masking RFC.
# Usage: LLVM_SOURCE_ROOT=/path/to/llvm-project BIN=/path/to/build/bin ./run.sh
set -euo pipefail
: "${BIN:?Set BIN to the llvm-project build/bin directory}"
: "${LLVM_SOURCE_ROOT:?Set LLVM_SOURCE_ROOT to the llvm-project source checkout}"
HERE="$(cd "$(dirname "$0")" && pwd)"

for tool in mlir-opt mlir-translate llc; do
  if [[ ! -x "$BIN/$tool" ]]; then
    printf 'error: %s is missing or not executable\n' "$BIN/$tool" >&2
    exit 1
  fi
done
if [[ ! -d "$LLVM_SOURCE_ROOT/llvm" || ! -d "$LLVM_SOURCE_ROOT/mlir" ]]; then
  printf 'error: LLVM_SOURCE_ROOT must point to an llvm-project source checkout\n' >&2
  exit 1
fi
LLVM_HEAD="$(git -C "$LLVM_SOURCE_ROOT" rev-parse HEAD)"

GPU_PIPE='builtin.module(func.func(convert-vector-to-gpu{use-nvgpu=true},canonicalize,cse))'
LLVM_PIPE='builtin.module(canonicalize,convert-vector-to-llvm,finalize-memref-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)'
KERN_PIPE='builtin.module(gpu.module(convert-vector-to-llvm,convert-gpu-to-nvvm,reconcile-unrealized-casts))'
TENSORCORE_PIPE='builtin.module(gpu.module(convert-vector-to-gpu{use-nvgpu=true},affine-expand-index-ops,lower-affine,convert-nvgpu-to-nvvm,convert-gpu-to-nvvm,convert-vector-to-llvm,convert-arith-to-llvm,reconcile-unrealized-casts,canonicalize,cse))'

echo "== Provenance =="
printf 'llvm-project HEAD: %s\n' "$LLVM_HEAD"
stat -c '%n: %y' "$BIN/mlir-opt" "$BIN/mlir-translate" "$BIN/llc"
echo

count_ptx () { # <ptx file> <name regex>
  python3 - "$1" "$2" <<'PY'
import re,sys
txt=open(sys.argv[1]).read()
parts=re.split(r'\n(?=\.visible \.entry|\.entry |\.visible \.func|\.func )', txt)
print(f"{'function':<40}{'instr':>6}{'bra':>5}{'loads':>7}")
for part in parts:
    m=re.search(sys.argv[2], part.split("\n")[0])
    if not m: continue
    body=[l for l in part.split("\n") if re.match(r'^\s+[a-z@]', l)]
    loads=len(re.findall(r'\bld\.(?:global\.)?b32', part))
    print(f"{m.group(1):<40}{len(body):>6}{len(re.findall(r'bra', part)):>5}{loads:>7}")
PY
}

echo "== A: what reaches the Tensor Core path? (dynamic shapes) =="
for f in A1_tensorcore_in_bounds A2_tensorcore_masked A3_tensorcore_no_in_bounds; do
  out="$("$BIN/mlir-opt" "$HERE/$f.mlir" -pass-pipeline="$GPU_PIPE")"
  printf '%-32s ldmatrix=%s mma.sync=%s transfer_read_left=%s contract_left=%s\n' "$f" \
    "$(grep -c 'nvgpu.ldmatrix'      <<<"$out" || true)" \
    "$(grep -c 'nvgpu.mma.sync'      <<<"$out" || true)" \
    "$(grep -c 'vector.transfer_read'<<<"$out" || true)" \
    "$(grep -c 'vector.contract'     <<<"$out" || true)"
done

echo
echo "== A/PTX: positive Tensor Core arm through mlir-opt -> mlir-translate -> llc =="
"$BIN/mlir-opt" "$HERE/A1_tensorcore_kernel.mlir" -pass-pipeline="$TENSORCORE_PIPE" \
  | python3 "$HERE/unwrap_gpu_module.py" > "$HERE/A1_tensorcore_kernel.llvm.mlir"
"$BIN/mlir-translate" "$HERE/A1_tensorcore_kernel.llvm.mlir" --mlir-to-llvmir \
  > "$HERE/A1_tensorcore_kernel.ll"
"$BIN/llc" -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80 \
  "$HERE/A1_tensorcore_kernel.ll" -o "$HERE/A1_tensorcore_kernel.ptx"
printf 'A1_tensorcore_kernel            ldmatrix=%s mma.sync=%s\n' \
  "$(grep -c 'ldmatrix\.sync' "$HERE/A1_tensorcore_kernel.ptx" || true)" \
  "$(grep -c 'mma\.sync'      "$HERE/A1_tensorcore_kernel.ptx" || true)"

echo
echo "== B: single 8xf32 load, minimal cases =="
for f in B1_load_in_bounds B2_load_masked_dynamic B3_load_masked_alltrue; do
  "$BIN/mlir-opt" "$HERE/$f.mlir" -pass-pipeline="$LLVM_PIPE" \
    | "$BIN/mlir-translate" --mlir-to-llvmir > "$HERE/$f.ll"
  "$BIN/llc" -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80 "$HERE/$f.ll" -o "$HERE/$f.ptx"
done
cat "$HERE"/B1_load_in_bounds.ptx "$HERE"/B2_load_masked_dynamic.ptx "$HERE"/B3_load_masked_alltrue.ptx > "$HERE/B_all.ptx"
count_ptx "$HERE/B_all.ptx" '(load_[A-Za-z0-9_]*)'

echo
echo "== C: full read/write matrix (4xf32) =="
"$BIN/mlir-opt" "$HERE/C_transfer_matrix.mlir" -pass-pipeline="$LLVM_PIPE" \
  | "$BIN/mlir-translate" --mlir-to-llvmir > "$HERE/C_transfer_matrix.ll"
"$BIN/llc" -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80 "$HERE/C_transfer_matrix.ll" -o "$HERE/C_transfer_matrix.ptx"
count_ptx "$HERE/C_transfer_matrix.ptx" '((?:read|write)_[A-Za-z0-9_]*)'

echo
echo "== D: real kernel ABI, global address space =="
"$BIN/mlir-opt" "$HERE/D_kernel_abi.mlir" -pass-pipeline="$KERN_PIPE" \
  | python3 "$HERE/unwrap_gpu_module.py" > "$HERE/D_kernel_abi.llvm.mlir"
"$BIN/mlir-translate" "$HERE/D_kernel_abi.llvm.mlir" --mlir-to-llvmir > "$HERE/D_kernel_abi.ll"
"$BIN/llc" -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80 "$HERE/D_kernel_abi.ll" -o "$HERE/D_kernel_abi.ptx"
count_ptx "$HERE/D_kernel_abi.ptx" '(kernel_[A-Za-z0-9_]*)'

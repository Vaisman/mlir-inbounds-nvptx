#!/usr/bin/env bash
# Masks that stay after peeling: which GPU conversions survive them.
# Usage: BIN=/path/to/build/bin ./run_masked_remainders.sh
set -euo pipefail
: "${BIN:?Set BIN to the llvm-project build/bin directory}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$(mktemp -d)"
OPT="$BIN/mlir-opt"

count () { grep -c -- "$1" "$2" 2>/dev/null || true; }
masks () { # <file>
  printf 'contract=%s masked_contract=%s vector.mask=%s' \
    "$(count 'vector.contract' "$1")" \
    "$(count 'vector.mask .*vector.contract' "$1")" \
    "$(count 'vector.mask' "$1")"
}
convert () { # <label> <input> <pass options>
  local out="$OUT/$1.mlir" err="$OUT/$1.err"
  if "$OPT" "$2" -convert-vector-to-gpu$3 -o "$out" 2>"$err"; then
    printf '%-18s exit=0 mma.sync=%s subgroup_mma_compute=%s vector.mask=%s\n' "$1" \
      "$(count 'nvgpu.mma.sync' "$out")" \
      "$(count 'gpu.subgroup_mma_compute' "$out")" \
      "$(count 'vector.mask' "$out")"
  else
    printf '%-18s exit=1 %s\n' "$1" "$(grep -m1 -oE "error: .*" "$err")"
  fi
}

echo "mlir-opt: $OPT"
echo "  mtime: $(stat -c %y "$OPT")"
echo "  md5:   $(md5sum "$OPT" | cut -d' ' -f1)"
for c in H_peeled_matmul I_peeled_matmul_wmma; do
  "$OPT" "$HERE/$c.mlir" -transform-interpreter -o "$OUT/$c.vec.mlir"
  "$OPT" "$OUT/$c.vec.mlir" -canonicalize -cse -o "$OUT/$c.canon.mlir"
  "$OPT" "$OUT/$c.canon.mlir" -test-eliminate-vector-masks=fixed-size -canonicalize -cse \
    -o "$OUT/$c.elim.mlir"
  echo "== $c"
  echo "after vectorize:            $(masks "$OUT/$c.vec.mlir")"
  echo "after canonicalize:         $(masks "$OUT/$c.canon.mlir")"
  echo "after eliminateVectorMasks: $(masks "$OUT/$c.elim.mlir")"
done
convert H_nvgpu "$OUT/H_peeled_matmul.elim.mlir" '=use-nvgpu=true'
convert I_subgroup_mma "$OUT/I_peeled_matmul_wmma.elim.mlir" ''
echo "outputs: $OUT"

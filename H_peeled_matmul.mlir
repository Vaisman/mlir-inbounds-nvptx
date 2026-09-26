// A dynamic matmul tiled for mma.sync (16x8x16), peeled in all three loops and
// vectorized with masking. The main loop has static tiles; the seven
// remainder loops keep genuine partial masks.
func.func @matmul_dyn(%A: memref<?x?xf16>, %B: memref<?x?xf16>, %C: memref<?x?xf16>) {
  linalg.matmul ins(%A, %B : memref<?x?xf16>, memref<?x?xf16>) outs(%C : memref<?x?xf16>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %li, %lj, %lk = transform.structured.tile_using_for %mm tile_sizes [16, 8, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">)
    %mk, %rk = transform.loop.peel %lk : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    %mj, %rj = transform.loop.peel %lj : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    %mi, %ri = transform.loop.peel %li : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    %all = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %all vector_sizes [16, 8, 16] create_named_contraction : !transform.any_op
    transform.yield
  }
}

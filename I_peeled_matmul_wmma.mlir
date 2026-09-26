// Vendor-neutral variant for gpu.subgroup_mma: static row strides, a dynamic M
// and B stored as (n, k). The M loop is peeled; the remainder keeps masks.
func.func @matmul_dyn_m(%A: memref<?x32xf16>, %B: memref<64x32xf16>, %C: memref<?x64xf16>) {
  linalg.matmul indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                                 affine_map<(m, n, k) -> (n, k)>,
                                 affine_map<(m, n, k) -> (m, n)>]
    ins(%A, %B : memref<?x32xf16>, memref<64x32xf16>) outs(%C : memref<?x64xf16>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %li, %lj, %lk = transform.structured.tile_using_for %mm tile_sizes [16, 16, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">)
    %mi, %ri = transform.loop.peel %li : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    %all = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %all vector_sizes [16, 16, 16] create_named_contraction : !transform.any_op
    transform.yield
  }
}

#rowcol = affine_map<(d0, d1) -> (d1, d0)>
#mA = affine_map<(d0, d1, d2) -> (d0, d2)>
#mB = affine_map<(d0, d1, d2) -> (d1, d2)>
#mC = affine_map<(d0, d1, d2) -> (d0, d1)>
// Mask-free diagnostic control: even if A2's mask is proved all-true and
// eliminated, these dynamic memref extents do not prove the tile is in bounds.
func.func @dyn_no_in_bounds(
    %a: memref<?x?xf16, #gpu.address_space<workgroup>>,
    %b: memref<?x?xf16, #gpu.address_space<workgroup>>,
    %c: memref<?x?xf16, #gpu.address_space<workgroup>>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.000000e+00 : f16
  %A = vector.transfer_read %a[%c0, %c0], %f0
      : memref<?x?xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
  %B = vector.transfer_read %b[%c0, %c0], %f0
      {permutation_map = #rowcol}
      : memref<?x?xf16, #gpu.address_space<workgroup>>, vector<8x16xf16>
  %C = vector.transfer_read %c[%c0, %c0], %f0
      : memref<?x?xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
  %D = vector.contract {
      indexing_maps = [#mA, #mB, #mC],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>} %A, %B, %C
      : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
  vector.transfer_write %D, %c[%c0, %c0]
      : vector<16x8xf16>, memref<?x?xf16, #gpu.address_space<workgroup>>
  return
}

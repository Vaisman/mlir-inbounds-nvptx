#rowcol = affine_map<(d0, d1) -> (d1, d0)>
#mA = affine_map<(d0, d1, d2) -> (d0, d2)>
#mB = affine_map<(d0, d1, d2) -> (d1, d2)>
#mC = affine_map<(d0, d1, d2) -> (d0, d1)>
// Same tile, boundedness expressed by masking instead of in_bounds.
func.func @dyn_masked(%a: memref<?x?xf16, #gpu.address_space<workgroup>>,
                      %b: memref<?x?xf16, #gpu.address_space<workgroup>>,
                      %c: memref<?x?xf16, #gpu.address_space<workgroup>>,
                      %m: index, %n: index, %k: index) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.000000e+00 : f16
  %maskA = vector.create_mask %m, %k : vector<16x16xi1>
  %maskB = vector.create_mask %k, %n : vector<16x8xi1>
  %maskC = vector.create_mask %m, %n : vector<16x8xi1>
  %A = vector.transfer_read %a[%c0, %c0], %f0, %maskA {in_bounds = [true, true]} : memref<?x?xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
  %B = vector.transfer_read %b[%c0, %c0], %f0, %maskB {permutation_map = #rowcol, in_bounds = [true, true]} : memref<?x?xf16, #gpu.address_space<workgroup>>, vector<8x16xf16>
  %C = vector.transfer_read %c[%c0, %c0], %f0, %maskC {in_bounds = [true, true]} : memref<?x?xf16, #gpu.address_space<workgroup>>, vector<16x8xf16>
  %D = vector.contract {indexing_maps = [#mA, #mB, #mC], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
  vector.transfer_write %D, %c[%c0, %c0], %maskC {in_bounds = [true, true]} : vector<16x8xf16>, memref<?x?xf16, #gpu.address_space<workgroup>>
  return
}

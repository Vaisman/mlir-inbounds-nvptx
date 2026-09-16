// Control: mask is statically all-true. Does MLIR recover the plain load?
func.func @load_mask_alltrue(%a: memref<?xf32>, %i: index) -> vector<8xf32> {
  %f0 = arith.constant 0.000000e+00 : f32
  %c8 = arith.constant 8 : index
  %mask = vector.create_mask %c8 : vector<8xi1>
  %v = vector.transfer_read %a[%i], %f0, %mask {in_bounds = [true]} : memref<?xf32>, vector<8xf32>
  return %v : vector<8xf32>
}

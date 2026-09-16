func.func @load_in_bounds(%a: memref<?xf32>, %i: index) -> vector<8xf32> {
  %f0 = arith.constant 0.000000e+00 : f32
  %v = vector.transfer_read %a[%i], %f0 {in_bounds = [true]} : memref<?xf32>, vector<8xf32>
  return %v : vector<8xf32>
}

func.func @load_masked(%a: memref<?xf32>, %i: index, %n: index) -> vector<8xf32> {
  %f0 = arith.constant 0.000000e+00 : f32
  %mask = vector.create_mask %n : vector<8xi1>
  %v = vector.transfer_read %a[%i], %f0, %mask {in_bounds = [true]} : memref<?xf32>, vector<8xf32>
  return %v : vector<8xf32>
}

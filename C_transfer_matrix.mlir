// Local RFC experiment. This file is intentionally not an LLVM regression test.
module attributes {llvm.target_triple = "nvptx64-nvidia-cuda"} {
  func.func @read_in_bounds(%src: memref<?xf32>, %i: index,
                            %pad: f32) -> vector<4xf32> {
    %v = vector.transfer_read %src[%i], %pad {in_bounds = [true]}
        : memref<?xf32>, vector<4xf32>
    return %v : vector<4xf32>
  }

  func.func @read_maybe_oob(%src: memref<?xf32>, %i: index,
                            %pad: f32) -> vector<4xf32> {
    %v = vector.transfer_read %src[%i], %pad
        : memref<?xf32>, vector<4xf32>
    return %v : vector<4xf32>
  }

  func.func @read_explicit_mask_in_bounds(
      %src: memref<?xf32>, %i: index, %n: index,
      %pad: f32) -> vector<4xf32> {
    %mask = vector.create_mask %n : vector<4xi1>
    %v = vector.transfer_read %src[%i], %pad, %mask {in_bounds = [true]}
        : memref<?xf32>, vector<4xf32>
    return %v : vector<4xf32>
  }

  func.func @read_explicit_mask_maybe_oob(
      %src: memref<?xf32>, %i: index, %n: index,
      %pad: f32) -> vector<4xf32> {
    %mask = vector.create_mask %n : vector<4xi1>
    %v = vector.transfer_read %src[%i], %pad, %mask
        : memref<?xf32>, vector<4xf32>
    return %v : vector<4xf32>
  }

  func.func @read_all_true_mask_in_bounds(
      %src: memref<?xf32>, %i: index,
      %pad: f32) -> vector<4xf32> {
    %mask = vector.constant_mask [4] : vector<4xi1>
    %v = vector.transfer_read %src[%i], %pad, %mask {in_bounds = [true]}
        : memref<?xf32>, vector<4xf32>
    return %v : vector<4xf32>
  }

  func.func @read_partial_constant_mask_in_bounds(
      %src: memref<?xf32>, %i: index,
      %pad: f32) -> vector<4xf32> {
    %mask = vector.constant_mask [3] : vector<4xi1>
    %v = vector.transfer_read %src[%i], %pad, %mask {in_bounds = [true]}
        : memref<?xf32>, vector<4xf32>
    return %v : vector<4xf32>
  }

  func.func @write_in_bounds(%v: vector<4xf32>, %dst: memref<?xf32>,
                             %i: index) {
    vector.transfer_write %v, %dst[%i] {in_bounds = [true]}
        : vector<4xf32>, memref<?xf32>
    return
  }

  func.func @write_maybe_oob(%v: vector<4xf32>, %dst: memref<?xf32>,
                             %i: index) {
    vector.transfer_write %v, %dst[%i]
        : vector<4xf32>, memref<?xf32>
    return
  }
}

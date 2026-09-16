// Local RFC experiment using NVPTX global-memory pointers and kernel ABI.
module attributes {gpu.container_module, llvm.target_triple = "nvptx64-nvidia-cuda"} {
  gpu.module @kernels [#nvvm.target<chip = "sm_80">] {
  gpu.func @kernel_in_bounds(%src: memref<1024xf32, 1>,
                              %dst: memref<4xf32, 1>, %i: index,
                              %pad: f32) kernel {
    %c0 = arith.constant 0 : index
    %v = vector.transfer_read %src[%i], %pad {in_bounds = [true]}
        : memref<1024xf32, 1>, vector<4xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]}
        : vector<4xf32>, memref<4xf32, 1>
    gpu.return
  }

  gpu.func @kernel_maybe_oob(%src: memref<1024xf32, 1>,
                              %dst: memref<4xf32, 1>, %i: index,
                              %pad: f32) kernel {
    %c0 = arith.constant 0 : index
    %v = vector.transfer_read %src[%i], %pad
        : memref<1024xf32, 1>, vector<4xf32>
    vector.transfer_write %v, %dst[%c0] {in_bounds = [true]}
        : vector<4xf32>, memref<4xf32, 1>
    gpu.return
  }
  }
}

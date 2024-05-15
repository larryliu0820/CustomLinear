//
//  ATenQuantizedLinear.m
//  CustomLinear
//
//  Created by Mengwei Liu on 5/10/24.
//

#include <torch/extension.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_weight_int4pack_mm_native.h>
#include <ATen/ops/_weight_int8pack_mm_native.h>
#include <ATen/ops/empty.h>
#endif
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>

// #define _CAPTURE_KERNEL 1

namespace at::native {

using namespace mps;

static char* QUANTIZED_KERNEL = R"METAL_QUANTIZED(
#include <metal_stdlib>
using namespace metal;

template<typename T>
struct Vec4Type {};

template<>
struct Vec4Type<float> {
  using type = float4;
};

template<>
struct Vec4Type<half> {
  using type = half4;
};

#if __METAL_VERSION__ >= 310
template<>
struct Vec4Type<bfloat> {
  using type = bfloat4;
};
#endif


// A is sizes.x x sizes.y
// B.T is sizes.z x sizes.y
// C is sizes.x x sizes.z
template<typename T>
kernel void int8pack_mm(
    constant T                 * A              [[buffer(0)]],
    constant char              * B              [[buffer(1)]],
    constant T                 * scales         [[buffer(2)]],
    device   T                 * outputData     [[buffer(3)]],
    constant uint3             & sizes          [[buffer(4)]],
    uint                         thread_index   [[thread_position_in_grid]]) {
    const uint lda = sizes.y;
    const uint ldc = sizes.z;
    const uint m = thread_index / sizes.z; // 0..sizes.x-1
    const uint n = thread_index % sizes.z; // 0..sizes.z-1
    using vecT = typename Vec4Type<T>::type;
    constant vecT *A_ptr = reinterpret_cast<constant vecT*>(A + m * lda);
    constant char4 *B_ptr = reinterpret_cast<constant char4*>(B + n * lda);

    float rc = 0.0;
    for(uint k = 0; k < sizes.y/4;  k++) {
      const auto a_val = float4(A_ptr[k]);
      const auto b_val = float4(B_ptr[k]);
      rc += dot(a_val, b_val);
    }
    outputData[thread_index] = T(rc * float(scales[n]));
}

#define INSTANTIATE_INT8MM(DTYPE)                                     \
template                                                              \
[[host_name("int8pack_mm_" #DTYPE)]]                                  \
kernel void int8pack_mm<DTYPE>(                                       \
    constant DTYPE            * A            [[buffer(0)]],           \
    constant char             * B            [[buffer(1)]],           \
    constant DTYPE            * scales       [[buffer(2)]],           \
    device   DTYPE            * outputData   [[buffer(3)]],           \
    constant uint3            & sizes        [[buffer(4)]],           \
    uint                        thread_index [[thread_position_in_grid]])

INSTANTIATE_INT8MM(half);
INSTANTIATE_INT8MM(float);
#if __METAL_VERSION__ >= 310
INSTANTIATE_INT8MM(bfloat);
#endif

)METAL_QUANTIZED";

Tensor _weight_int8pack_mm_mps(const Tensor& A, const Tensor& B, const Tensor& scales) {
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);

  TORCH_CHECK(A.dtype() == kBFloat16 || A.dtype() == kHalf || A.dtype() == kFloat,
              __func__,
              " : expect A to be either 32-bit or 16-bit float tensor.");
  TORCH_CHECK(A.is_contiguous(), __func__, " : expect A to be contiguous.");
  TORCH_CHECK(A.dim() == 2, __func__, " : expect A to be 2D tensor.");

  TORCH_CHECK(B.dtype() == kChar, __func__, " : expect B to be int8 tensor.");
  TORCH_CHECK(B.is_contiguous(), __func__, " : expect B to be contiguous.");
  TORCH_CHECK(B.size(1) == K, __func__, " : expect B.size(1) == ", K);

  TORCH_CHECK(scales.dim() == 1 && scales.size(0) == N, __func__, " : expect scales to be 1d tensor with size ", N);

  auto C = at::empty({M, N}, A.options());
  MPSStream* mpsStream = getCurrentMPSStream();
  std::array<uint32_t, 3> sizes = {static_cast<uint32_t>(M), static_cast<uint32_t>(K), static_cast<uint32_t>(N)};
  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      NSError *error = nil;
#if _CAPTURE_KERNEL
      if (getMPSProfiler().isCaptureEnabled()) {
        getMPSProfiler().startCapture(__func__, mpsStream);
      }
#endif
      id<MTLDevice> device = mpsStream->device();
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      // Load the custom linear shader
      id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:QUANTIZED_KERNEL]
                                                                  options:nil
                                                                    error:&error];
      TORCH_CHECK(customKernelLibrary, "Error creating custom kernel library: ", error.localizedDescription.UTF8String);
      const std::string kernel = "int8pack_mm_" + scalarToMetalTypeString(A.scalar_type());

      // Create a function
      id<MTLFunction> customQuantizedLinearFunction = [customKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
      TORCH_CHECK(customQuantizedLinearFunction, "Error creating custom kernel function: ", kernel);

      id<MTLComputePipelineState> quantizedPSO = [device newComputePipelineStateWithFunction:customQuantizedLinearFunction error:&error];
      TORCH_CHECK(quantizedPSO, error.localizedDescription.UTF8String);

      [computeEncoder setComputePipelineState:quantizedPSO];
      mtl_setBuffer(computeEncoder, A, 0);
      mtl_setBuffer(computeEncoder, B, 1);
      mtl_setBuffer(computeEncoder, scales, 2);
      mtl_setBuffer(computeEncoder, C, 3);
      [computeEncoder setBytes:sizes.data() length:sizeof(uint32_t) * sizes.size() atIndex:4];
      mtl_dispatch1DJob(computeEncoder, quantizedPSO, C.numel());
#if _CAPTURE_KERNEL
      if (getMPSProfiler().isCapturing()) {
        getMPSProfiler().stopCapture(mpsStream);
      }
#endif
    }
  });

  return C;
}

} // namespace at::native

// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mps_int8_linear", &at::native::_weight_int8pack_mm_mps);
}

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
#define BLOCK_SIZE_M 32 // 4 simdgroup matrices from matrix A
#define BLOCK_SIZE_N 64 // 8 simdgroup matrices from matrix B
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 2 // each thread take 2 simdgroup matrices from matrix A
#define THREAD_MAT_N 4 // each thread take 4 simdgroup matrices from matrix B
#define THREAD_PER_ROW 2 // 2 thread for each row in matrix A to load numbers
#define THREAD_PER_COL 4 // 4 thread for each row in matrix B to load numbers
#define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8
#define SG_MAT_ROW 8

template <typename type4x4>
void dequantize_q8_0(constant char *weight, constant float *scales, thread type4x4 & reg) {
    constant int8_t * qs = (constant const int8_t *)weight;

    for (int i = 0; i < 16; i++) {
        reg[i/4][i%4] = qs[i] * scales[i];
    }
}

// 2 x 4096 @ 1024 x 4096 * 1024 -> x.transpose(weight) * scales -> A.transpose(B) * scales
// M x K @ N x K -> M x N
[[host_name("int8pack_mm_float")]]
kernel void int8pack_mm(
    constant float             * A              [[buffer(0)]],  // 2 x 4096
    constant char              * B              [[buffer(1)]],  // 1024 x 4096
    constant float             * scales         [[buffer(2)]],  // 1024
    device   float             * outputData     [[buffer(3)]],  // 2 x 1024
    constant uint3             & sizes          [[buffer(4)]],
    threadgroup uchar          * shared_memory  [[threadgroup(0)]], // threadgroup buffer at index 0
    uint3                        tgpig          [[threadgroup_position_in_grid]], // 3d coordinates
    uint                         tiitg          [[thread_index_in_threadgroup]], // 128 per threadgroup
    uint                         sgitg          [[simdgroup_index_in_threadgroup]]) { // what is this
    
    int64_t ne00 = sizes.x;
    int64_t ne01 = sizes.y;
    int64_t nb00 = sizeof(float);
    int64_t nb01 = nb00 * ne00;
    int64_t nb02 = nb01 * ne01;
    int64_t ne10 = sizes.z;
    int64_t ne11 = sizes.y;
    int64_t nb10 = sizeof(char);
    int64_t nb11 = nb10 * ne10;
    int64_t nb12 = nb11 * ne11;
    int64_t ne0 = sizes.x;
    int64_t ne1 = sizes.z;
        
    // [encoder setThreadgroupMemoryLength:8192 atIndex:0]; threadgroup buffer has a length 8192.
    threadgroup float * sa = (threadgroup float *)(shared_memory); // first half.
    threadgroup float * sb = (threadgroup float *)(shared_memory + 4096); // second half.
        
    const uint r0 = tgpig.y;
    const uint r1 = tgpig.x;
    
    // if this block is of 32x64 shape or smaller
    short n_rows = (ne0 - r0 * BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0 * BLOCK_SIZE_M) : BLOCK_SIZE_M;
    short n_cols = (ne1 - r1 * BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1 * BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    /**
     tiitg represents the index of the current thread within the threadgroup.
     THREAD_PER_ROW is the number of threads assigned to load data from each row of the matrix.
     n_rows is the number of rows in the current block of the matrix.
     The expression (short)tiitg/THREAD_PER_ROW calculates which row the thread should load data from.
     However, to prevent threads from loading data outside of the matrix boundaries, the result is compared to n_rows. If the calculated row index exceeds n_rows, it's clamped to n_rows - 1.
     This ensures that each thread loads data only from within the valid range of rows in the matrix block.
     NOTE: this is a relative index in reference to BLOCK_SIZE_M. So when we index to the actual row we need to add r0 * BLOCK_SIZE_M first.
     */
    short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;
    /**
     Similar to the thread row calculation, this line calculates the column index for the current thread within the matrix block.
     THREAD_PER_COL represents the number of threads assigned to load data from each column of the matrix.
     n_cols is the number of columns in the current block of the matrix.
     The expression (short)tiitg/THREAD_PER_COL calculates which column the thread should load data from.
     The result is compared to n_cols to ensure that the thread does not load data outside the valid range of columns. If the calculated column index exceeds n_cols, it's clamped to n_cols - 1.
     NOTE: this is a relative index in reference to BLOCK_SIZE_N. So when we index to the actual row we need to add r1 * BLOCK_SIZE_N first.
     */
    simdgroup_float8x8 ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 c_res[8];
    for (int i = 0; i < 8; i++){
        c_res[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    /**
     Calculates which portion of the row this thread should take care of
     */
    constant float* x = (constant float*)((constant char*)A
        + nb01 * (r0 * BLOCK_SIZE_M + thread_row)
        + nb00 * (BLOCK_SIZE_K / THREAD_PER_ROW * (tiitg % THREAD_PER_ROW)));
    constant char* y = B
        + nb11 * (r1 * BLOCK_SIZE_N + thread_col)
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL));

    for (int loop_k = 0; loop_k < ne01; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        float4x4 temp_a;
        dequantize_q8_0(y, scales + loop_k, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (int i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8) \
            +                     (tiitg % THREAD_PER_ROW) * 16 + (i / 8) * 8) \
            +                     (tiitg / THREAD_PER_ROW) % 8  + (i & 7) * 8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) = *((constant float2x4 *)y);

        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup float * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
        threadgroup float * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));

        #pragma unroll(4)
        for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            #pragma unroll(4)
            for (int i = 0; i < 4; i++) {
                simdgroup_load(ma[i],lsma + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll(2)
            for (int i = 0; i < 2; i++) {
                simdgroup_load(mb[i],lsmb + SG_MAT_SIZE * i);
            }

            lsma += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
            lsmb += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;

            #pragma unroll(8)
            for (int i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(c_res[i], mb[i/4], ma[i%4], c_res[i]);
            }
        }
    }

    if ((r0 + 1) * BLOCK_SIZE_M <= ne0 && (r1 + 1) * BLOCK_SIZE_N <= ne1) {
        device float * C = outputData + (BLOCK_SIZE_M * r0 + 32 * (sgitg &  1)) \
                                      + (BLOCK_SIZE_N * r1 + 16 * (sgitg >> 1)) * ne0
                                      + ne1*ne0;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], C + 8 * (i%4) + 8 * ne0 * (i/4), ne0);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *)shared_memory) \
                                      + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_M;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_M * (i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float * C = outputData + (BLOCK_SIZE_M * r0) + (BLOCK_SIZE_N * r1) * ne0 + ne1*ne0;
        if (sgitg == 0) {
            for (int i = 0; i < n_rows; i++) {
                for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                    *(C + i + j * ne0) = *(temp_str + i + j * BLOCK_SIZE_M);
                }
            }
        }
    }
}
    
)METAL_QUANTIZED";

Tensor _llama_cpp_int8pack_mm_mps(const Tensor& A, const Tensor& B, const Tensor& scales) {
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);

  TORCH_CHECK(A.dtype() == kFloat,
              __func__,
              " : expect A to be either 32-bit float tensor.");
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
        [computeEncoder setThreadgroupMemoryLength:8192 atIndex:0];
        [computeEncoder dispatchThreadgroups:MTLSizeMake( (M + 31)/32, (N + 63)/64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
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
    m.def("llama_cpp_mps_int8_linear", &at::native::_llama_cpp_int8pack_mm_mps);
}
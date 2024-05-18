//
//  File.swift
//  CustomLinear
//
//  Created by Mengwei Liu on 5/16/24.
//

import Metal

let shader_source = """

#include <metal_stdlib>
using namespace metal;

template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    float4x4 temp = *(((device float4x4 *)src));
    for (int i = 0; i < 16; i++){
        reg[i/4][i%4] = temp[i/4][i%4];
    }
}


#define BLOCK_SIZE_M 64 // 8 simdgroup matrices from matrix A
#define BLOCK_SIZE_N 32 // 4 simdgroup matrices from matrix B
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 4 // each thread take 4 simdgroup matrices from matrix A
#define THREAD_MAT_N 2 // each thread take 2 simdgroup matrices from matrix B
#define THREAD_PER_ROW 2 // 2 thread for each row in matrix A to load numbers
#define THREAD_PER_COL 4 // 4 thread for each row in matrix B to load numbers
#define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8
#define SG_MAT_ROW 8

// each block_q contains 16*nl weights
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
void kernel_mul_mm_impl(device const  uchar * src0,
                        device const  uchar * src1,
                        device        float * dst,
                        constant    int64_t & ne00,
                        constant    int64_t & ne02,
                        constant   uint64_t & nb01,
                        constant   uint64_t & nb02,
                        constant    int64_t & ne12,
                        constant   uint64_t & nb10,
                        constant   uint64_t & nb11,
                        constant   uint64_t & nb12,
                        constant    int64_t & ne0,
                        constant    int64_t & ne1,
                        constant       uint & r2,
                        constant       uint & r3,
                        threadgroup   uchar * shared_memory [[threadgroup(0)]],
                        uint3                 tgpig[[threadgroup_position_in_grid]],
                        uint                  tiitg[[thread_index_in_threadgroup]],
                        uint                  sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup half  * sa = (threadgroup half  *)(shared_memory);
    threadgroup float * sb = (threadgroup float *)(shared_memory + 4096);

    const uint r0 = tgpig.y;
    const uint r1 = tgpig.x;
    const uint im = tgpig.z;

    // if this block is of 64x32 shape or smaller
    short n_rows = (ne0 - r0 * BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0 * BLOCK_SIZE_M) : BLOCK_SIZE_M;
    short n_cols = (ne1 - r1 * BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1 * BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_half8x8  ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 c_res[8];
    for (int i = 0; i < 8; i++){
        c_res[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    short il = (tiitg % THREAD_PER_ROW);

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    uint   offset0 = (i12/r2)*nb02 + (i13/r3)*(nb02*ne02);
    ushort offset1 = il/nl;

    device const block_q * x = (device const block_q *)(src0 + (r0 * BLOCK_SIZE_M + thread_row) * nb01 + offset0) + offset1;
    device const float   * y = (device const float   *)(src1
        + nb12 * im
        + nb11 * (r1 * BLOCK_SIZE_N + thread_col)
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        half4x4 temp_a;
        dequantize_func(x, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (int i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8) \
            +                     (tiitg % THREAD_PER_ROW) * 16 + (i / 8) * 8) \
            +                     (tiitg / THREAD_PER_ROW) % 8  + (i & 7) * 8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) = *((device float2x4 *)y);

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2+nl-1)/nl : x;
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup half  * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
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
        device float * C = dst + (BLOCK_SIZE_M * r0 + 32 * (sgitg &  1)) \
                               + (BLOCK_SIZE_N * r1 + 16 * (sgitg >> 1)) * ne0 + im*ne1*ne0;
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

        device float * C = dst + (BLOCK_SIZE_M * r0) + (BLOCK_SIZE_N * r1) * ne0 + im*ne1*ne0;
        if (sgitg == 0) {
            for (int i = 0; i < n_rows; i++) {
                for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                    *(C + i + j * ne0) = *(temp_str + i + j * BLOCK_SIZE_M);
                }
            }
        }
    }
}

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void kernel_mul_mm(device const  uchar * src0,
                          device const  uchar * src1,
                          device        float * dst,
                          constant    int64_t & ne00,
                          constant    int64_t & ne02,
                          constant   uint64_t & nb01,
                          constant   uint64_t & nb02,
                          constant    int64_t & ne12,
                          constant   uint64_t & nb10,
                          constant   uint64_t & nb11,
                          constant   uint64_t & nb12,
                          constant    int64_t & ne0,
                          constant    int64_t & ne1,
                          constant       uint & r2,
                          constant       uint & r3,
                          threadgroup   uchar * shared_memory [[threadgroup(0)]],
                          uint3                 tgpig[[threadgroup_position_in_grid]],
                          uint                  tiitg[[thread_index_in_threadgroup]],
                          uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mm_impl<block_q, nl, dequantize_func>(
        src0,
        src1,
        dst,
        ne00,
        ne02,
        nb01,
        nb02,
        ne12,
        nb10,
        nb11,
        nb12,
        ne0,
        ne1,
        r2,
        r3,
        shared_memory,
        tgpig,
        tiitg,
        sgitg);
}

typedef decltype(kernel_mul_mm<float4x4, 1, dequantize_f32>) mat_mm_t;

template [[host_name("kernel_mul_mm_f32_f32")]]
kernel mat_mm_t kernel_mul_mm<float4x4,      1,     dequantize_f32>;
"""

guard let device = MTLCopyAllDevices().first else { fatalError("Not Metal device found") }

print("Using device \(device.name)")

let options = MTLCompileOptions()
options.languageVersion = .version3_1
options.fastMathEnabled = false
let library = try! device.makeLibrary(source:shader_source, options:options)
guard let mfunc = library.makeFunction(name: "kernel_mul_mm_f32_f32") else { fatalError("Can't find function") }
// Create a command queue
let commandQueue = device.makeCommandQueue()
// Create a compute pipeline state
let pipelineState = try? device.makeComputePipelineState(function: mfunc)
// Create a command buffer
let commandBuffer = commandQueue?.makeCommandBuffer()

// Create a compute command encoder
let computeEncoder = commandBuffer?.makeComputeCommandEncoder()
// Prepare the data
let src0: [Float] = Array(repeating: 2.0, count: 4096) // weight, 64x64
let src1: [Float] = Array(repeating: 1.0, count: 2048) // input, 64x32
let dst: [Float] = Array(repeating: 0.0, count: 2048) // output, 64x32
// Create buffers
let src0Buffer = device.makeBuffer(bytes: src0, length: src0.count, options: [.storageModeShared])
let src1Buffer = device.makeBuffer(bytes: src1, length: src1.count, options: [.storageModeShared])
let dstBuffer = device.makeBuffer(bytes: dst, length: dst.count * MemoryLayout<Float>.size, options: [.storageModeShared])
// Set the buffers
computeEncoder?.setBuffer(src0Buffer, offset: 0, index: 0)
computeEncoder?.setBuffer(src1Buffer, offset: 0, index: 1)
computeEncoder?.setBuffer(dstBuffer, offset: 0, index: 2)
// Set the constants
let constants: [Int64] = [64, 1, 256, 16384, 1, 4, 256, 8192, 64, 32]
for (index, constant) in constants.enumerated() {
    var value = constant
    computeEncoder?.setBytes(&value, length: MemoryLayout<Int64>.size, index: index + 3)
}
// Set the uint constants
let uintConstants: [UInt32] = [1, 1]
for (index, constant) in uintConstants.enumerated() {
    var value = constant
    computeEncoder?.setBytes(&value, length: MemoryLayout<UInt32>.size, index: index + constants.count + 3)
}
// Set the threadgroup memory length
computeEncoder?.setThreadgroupMemoryLength(8192, index: 0)
// Set the compute pipeline state
computeEncoder?.setComputePipelineState(pipelineState!)
// Dispatch the threads
let threadGroupSize = MTLSize(width: 128, height: 1, depth: 1)
let threadGroups = MTLSize(width: 1, height: 1, depth: 1)
computeEncoder?.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
let captureManager = MTLCaptureManager.shared()
// Create a capture descriptor
let captureDescriptor = MTLCaptureDescriptor()
guard captureManager.supportsDestination(.gpuTraceDocument) else {
    print("Capturing to a GPU trace file isn't supported.")
    exit(1)
}
captureDescriptor.captureObject = device
captureDescriptor.destination = .gpuTraceDocument
captureDescriptor.outputURL = URL(fileURLWithPath: "/Users/larryliu/Desktop/llama_cpp_float_linear.gputrace")

do {
    try captureManager.startCapture(with: captureDescriptor)
} catch {
    print("Failed to start capture, error: \(error)")
}
// End encoding
computeEncoder?.endEncoding()
// Commit the command buffer
commandBuffer?.commit()
// Wait for the command buffer to complete execution
commandBuffer?.waitUntilCompleted()
captureManager.stopCapture()

let resultData = NSData(bytesNoCopy: dstBuffer!.contents(),
                        length: dstBuffer!.length,
                        freeWhenDone: false)
var resultArray = [Float](repeating: 0, count: resultData.length)
resultData.getBytes(&resultArray, length: resultData.length)
print(resultArray)

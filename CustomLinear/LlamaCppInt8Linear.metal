//
//  LlamaCppInt8Linear.metal
//  CustomLinear
//
//  Created by Mengwei Liu on 5/13/24.
//


#include <metal_stdlib>
using namespace metal;
#define BLOCK_SIZE_M 32 // 4 simdgroup matrices from matrix A
#define BLOCK_SIZE_N 64 // 8 simdgroup matrices from matrix B
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 2 // each thread take 2 simdgroup matrices from matrix A
#define THREAD_MAT_N 4 // each thread take 4 simdgroup matrices from matrix B
#define THREAD_PER_ROW 4 // 4 thread for each row in matrix A to load numbers. This needs to be 128 / BLOCK_SIZE_M
#define THREAD_PER_COL 2 // 2 thread for each row in matrix B to load numbers. This needs to be 128 / BLOCK_SIZE_N
#define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8
#define SG_MAT_ROW 8

template <typename type4x4>
void dequantize_q8_0(constant char *weight, constant float *scales, thread type4x4 & reg) {
    constant int8_t * qs = (constant const int8_t *)weight;

    for (int i = 0; i < 16; i++) {
        reg[i/4][i%4] = qs[i] * scales[0];
    }
}

// 2 x 4096 @ 1024 x 4096 * 1024 -> x.transpose(weight) * scales -> A.transpose(B) * scales
// M x K @ N x K -> M x N
[[host_name("mengwei_mm")]]
kernel void int8pack_mm(
    constant float             * A              [[buffer(0)]],  // 2 x 4096
    constant char              * B              [[buffer(1)]],  // 1024 x 4096
    constant float             * scales         [[buffer(2)]],  // 1024
    device   float             * outputData     [[buffer(3)]],  // 2 x 1024
    constant uint3             & sizes          [[buffer(4)]],
    threadgroup uchar          * shared_memory  [[threadgroup(0)]], // threadgroup buffer at index 0
    uint3                        tgpig          [[threadgroup_position_in_grid]], // 3d coordinates
    uint                         tiitg          [[thread_index_in_threadgroup]], // 128 per threadgroup
    uint                         sgitg          [[simdgroup_index_in_threadgroup]]) {
    
    uint32_t ne00 = sizes.x;
    uint32_t ne01 = sizes.y;
    uint32_t nb00 = sizeof(float);
    uint32_t nb01 = nb00 * ne00;
    uint32_t nb02 = nb01 * ne01;
    uint32_t ne10 = sizes.z;
    uint32_t ne11 = sizes.y;
    uint32_t nb10 = sizeof(char);
    uint32_t nb11 = nb10 * ne10;
    uint32_t nb12 = nb11 * ne11;
    uint32_t ne0 = sizes.x;
    uint32_t ne1 = sizes.z;
        
    // [encoder setThreadgroupMemoryLength:8192 atIndex:0]; threadgroup buffer has a length 8192.
    threadgroup half * sa = (threadgroup half *)(shared_memory); // first half for storing weight.
    threadgroup float * sb = (threadgroup float *)(shared_memory + 4096); // second half for storing input.
        
    const uint r0 = tgpig.y;
    const uint r1 = tgpig.x;
    
    // if this block is of 32x64 shape or smaller,
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
    // Need to use these N times to calculate 1 block of result BLOCK_SIZE_M x BLOCK_SIZE_N
    simdgroup_half8x8 ma[4];
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

    for (uint loop_k = 0; loop_k < ne01; loop_k += BLOCK_SIZE_K) { // BLOCK_SIZE_K = 32
        // load data and store to threadgroup memory. Now we've partitioned weight to be BLOCK_SIZE_N x BLOCK_SIZE_K
        // Input to be BLOCK_SIZE_M x BLOCK_SIZE_K, need to fill them both into shared memory.
        //
        // dequantize the weight into half. The current thread only need to care about 16 weights because
        // BLOCK_SIZE_K / THREAD_PER_COL = 16
        half4x4 temp_a;
        
        // find the scale index
        int scale_index = r1 * BLOCK_SIZE_N + thread_col;
        dequantize_q8_0(y, scales + scale_index, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // doing a transpose. Currently the weight block is BLOCK_SIZE_N x BLOCK_SIZE_K (64 x 32) we need to map that
        // to transposed 8x8 blocks, in contiguous memory format. The grid of 8x8 should be 4x8 (64x32 transposed).
        #pragma unroll(16)
        for (int i = 0; i < 16; i++) {
            // calculate the sg mat index in 4x8 grid using tiitg and i, in contiguous format, meaning (1, 2) -> 6
            // notice that i can only affect 2 sg mat.
            
            // for example, tiitg 32, i 12 -> 0 + 1 = 1, it needs to work on sg mat grid row 1
            int sg_mat_grid_row_index = (tiitg % THREAD_PER_COL) * THREAD_PER_COL + i / 8;
            // same example, sg mat grid col index: 32 / 2 / 8 = 2, so currently need to work with sg mat at (1, 2)
            int sg_mat_grid_col_index = tiitg / THREAD_PER_COL / 8;
            // now inside sg mat, which index to write to? starting point is SG_MAT_SIZE * sg_mat_offset
            int row_offset = i & 7;
            int col_offset = (tiitg / THREAD_PER_COL) % 8;
            // now calculates the overall offset for sa
            int sa_offset = (sg_mat_grid_row_index * 8 + sg_mat_grid_col_index) * 64 + (row_offset * 8 + col_offset);
            // write data
            *(sa + sa_offset) = temp_a[i/4][i%4];
        }
        // copy 8 activation values
        int sb_offset = tiitg * (BLOCK_SIZE_K / THREAD_PER_ROW);
        *((threadgroup float2x4 *) sb + sb_offset) = *((constant float2x4 *)x);

        y += BLOCK_SIZE_K; // why?
        x += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // at this point all the 128 threads should finished storing data into shared memory.
        // load matrices from shared memory and conduct outer product form of matrix multiplication.
        // we start to use SG_MAT_SIZE and THREAD_MAT_M etc.
        //
        // 2 simdgroup in a threadgroup.
        threadgroup half * lsma = (sa + THREAD_MAT_N * SG_MAT_SIZE * (sgitg % 2));
        threadgroup float * lsmb = (sb + THREAD_MAT_M * SG_MAT_SIZE * (sgitg / 2));

        // For loop going through BLOCK_SIZE_K (32)
        #pragma unroll(4)
        for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            #pragma unroll(4)
            for (int i = 0; i < 4; i++) {
                // load 4 sg mat into ma.
                int a_offset = SG_MAT_SIZE * i;
                simdgroup_load(ma[i], lsma + a_offset);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll(2)
            for (int i = 0; i < 2; i++) {
                // load 2 sg mat into mb.
                int b_offset = SG_MAT_SIZE * i;
                simdgroup_load(mb[i], lsmb + b_offset);
            }
            // should be stepping THREAD_MAT_N * SG_MAT_SIZE * 2. It's just happens to be the same as BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE
            int step_N = BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;
            int step_M = BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
            
            lsma += step_N;
            lsmb += step_M;

            #pragma unroll(8)
            for (int i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(c_res[i], mb[i/4], ma[i%4], c_res[i]);
            }
        }
    }

    if ((r0 + 1) * BLOCK_SIZE_N <= ne0 && (r1 + 1) * BLOCK_SIZE_M <= ne1) {
        device float * C = outputData + (BLOCK_SIZE_N * r0 + 32 * (sgitg &  1)) \
                                      + (BLOCK_SIZE_M * r1 + 16 * (sgitg >> 1)) * ne0
                                      + ne1*ne0;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], C + 8 * (i%4) + 8 * ne0 * (i/4), ne0);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *)shared_memory) \
                                      + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_N;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_N * (i/4), BLOCK_SIZE_N);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float * C = outputData + (BLOCK_SIZE_N * r0) + (BLOCK_SIZE_M * r1) * ne0 + ne1*ne0;
        if (sgitg == 0) {
            for (int i = 0; i < n_rows; i++) {
                for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_M) {
                    *(C + i + j * ne0) = *(temp_str + i + j * BLOCK_SIZE_N);
                }
            }
        }
    }
}
    

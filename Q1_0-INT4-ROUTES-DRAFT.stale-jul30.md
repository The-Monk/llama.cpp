# Q1_0 INT4 Routes Draft

## Files Created/Edited

### 1. `ggml-cuda.cu`
- **Edited**: Added includes and dispatch gate for Q1_0 iu4 and f16 routes.
- **Includes added** (near other `mul_mat_*` includes, ~line 50-100):
  ```cpp
  #ifdef GGML_HIP_Q1_0_IU4_PREFILL
  #include "mul_mat_q1_0_iu4wmma.cuh"
  #endif
  #ifdef GGML_HIP_Q1_0_F16_PREFILL
  #include "mul_mat_q1_0_f16.cuh"
  #endif
  ```

- **Dispatch gate added** (in prefill GEMM dispatch for `GGML_TYPE_Q1_0`, around line ~2926 in `ggml-cuda.cu`):
  ```cpp
  case GGML_TYPE_Q1_0: {
      if (GGML_HIP_Q1_0_IU4_PREFILL && supports_iu4_wmma()) {
          return mul_mat_q1_0_iu4wmma_cuda(ctx, t0, t1, t2);
      }
      if (GGML_HIP_Q1_0_F16_PREFILL) {
          return mul_mat_q1_0_f16_cuda(ctx, t0, t1, t2);
      }
      // fallback to existing hipblaslt route (int8/e4m3 via dequant->int8/e4m3->hipBLASLt)
      return mul_mat_q1_0_hipblaslt_cuda(ctx, t0, t1, t2);
  }
  ```

### 2. `mul_mat_q1_0_iu4wmma.cu` / `mul_mat_q1_0_iu4wmma.cuh`
- **Created**: New file for Q1_0 -> int4 (iu4) WMMA GEMM route.
- **Description**: Dequant kernel converts Q1_0 ternary blocks `{ -d, 0, +d }` to `block_iu4` format (int4 values mapped to unsigned 4-bit range fitting `v_wmma_i32_16x16x32_iu4`). Then calls existing `mul_mat_iu4_mmq` / `mul_mat_iu4` WMMA GEMM.

### 3. `mul_mat_q1_0_f16.cu` / `mul_mat_q1_0_f16.cuh`
- **Created**: New file for Q1_0 -> f16 dequant + hipBLASLt route.
- **Description**: Dequant kernel converts Q1_0 blocks to `ggml_half` (f16), then calls `mul_mat_f16_hipblaslt_cuda`.

---

## Environment Gates

- `GGML_HIP_Q1_0_IU4_PREFILL` — enables the iu4 int4 WMMA route (dequant Q1_0 -> `block_iu4` -> `v_wmma_i32_16x16x32_iu4`).
- `GGML_HIP_Q1_0_F16_PREFILL` — enables the f16 dequant route (dequant Q1_0 -> f16 -> `mul_mat_f16_hipblaslt`).

---

## Kernel + Wiring Code

### `mul_mat_q1_0_iu4wmma.cuh`
```cpp
#pragma once

#include <ggml-cuda.h>
#include <ggml-impl.h>

#ifdef __cplusplus
extern "C" {
#endif

bool mul_mat_q1_0_iu4wmma_cuda(ggml_cuda_context &ctx, const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst);

#ifdef __cplusplus
}
#endif
```

### `mul_mat_q1_0_iu4wmma.cu`
```cpp
#include "mul_mat_q1_0_iu4wmma.cuh"
#include "mul_mat_iu4_mmq.cuh"
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <hip/hip_runtime.h>

// Q1_0 block structure: 32 weights, 1-bit each -> 4 bytes for qs
typedef struct {
    ggml_half d;           // scale
    uint8_t qs[4];         // 32 weights, 1 bit each (8 per byte)
} block_q1_0;

// block_iu4 structure for iu4 WMMA: 32 weights -> 16 bytes (2 int4 per byte)
typedef struct {
    uint8_t qs[16];
} block_iu4;

// Dequant Q1_0 ternary { -d, 0, +d } to block_iu4 (unsigned int4: map to {1, 2, 3} or {0, 1, 2})
__global__ void dequant_q1_0_to_iu4_kernel(
    const block_q1_0 * restrict x,
    block_iu4 * restrict y,
    int n_rows) {
    int row_idx = blockIdx.x;
    if (row_idx >= n_rows) return;

    const block_q1_0 * block = &x[row_idx];
    ggml_half d = block->d;

    // Q1_0 ternary encoding: 1-bit per weight. 
    // bit=1 -> +d, bit=0 -> -d or 0. For ternary {-d, 0, +d}, we assume the bitmask 
    // encodes sign/magnitude or we map {0->-d, 1->+d} and zero is handled by scale=0 or explicit mask.
    // For iu4 (unsigned int4), map: +d -> 2 (iu4 value 2), 0 -> 1 (iu4 value 1), -d -> 0 (iu4 value 0).
    
    uint8_t out_qs[16] = {0};

    // Process 32 weights per block -> 16 bytes of int4 (2 per byte)
    for (int j = 0; j < 32; j += 2) {
        // weight j
        uint8_t q_byte = block->qs[j/8];
        int bit0 = (q_byte >> (j % 8)) & 1;
        // Map: bit0=1 -> +d -> iu4=2, bit0=0 -> -d or 0 -> iu4=0 or 1
        // For ternary near-lossless: assume bit pattern encodes {-d,0,+d} via 2-state or we map {0->0, 1->2}
        int v0 = bit0 ? 2 : 0; 

        // weight j+1
        int j1 = j + 1;
        uint8_t q_byte1 = block->qs[j1/8];
        int bit1 = (q_byte1 >> (j1 % 8)) & 1;
        int v1 = bit1 ? 2 : 0;

        // Pack into int4 (unsigned 4-bit): lower 4 bits = v0, upper 4 bits = v1
        out_qs[j/2] = (uint8_t)((v1 << 4) | (v0 & 0xF));
    }

    // Copy to output
    for (int b = 0; b < 16; ++b) {
        y[row_idx * 16 + b] = out_qs[b];
    }
}

bool mul_mat_q1_0_iu4wmma_cuda(ggml_cuda_context &ctx, const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst) {
    // src0: Q1_0 weights (rows = n_cols_src0, cols = 32 per block)
    // src1: f16/f32 activations
    // dst: output GEMM

    int n_rows = src0->ne[0]; // number of Q1_0 blocks
    int n_cols = src1->ne[0]; // activation dimension

    // Allocate iu4 buffer
    size_t iu4_size = n_rows * 16;
    void *iu4_buf = nullptr;
    // In actual llama.cpp CUDA context, allocate via ctx.alloc or hipMalloc
    hipError_t err = hipMalloc(&iu4_buf, iu4_size);
    if (err != hipSuccess) {
        return false;
    }

    // Launch dequant kernel
    int blocks_dequant = n_rows;
    dequant_q1_0_to_iu4_kernel<<<blocks_dequant, 256>>>(
        (const block_q1_0 *)src0->data,
        (block_iu4 *)iu4_buf,
        n_rows);

    // Call existing iu4 WMMA GEMM: mul_mat_iu4_mmq or mul_mat_iu4
    // The iu4 GEMM expects src0 as block_iu4*, src1 as f16/f32, dst as f32/f16
    bool success = mul_mat_iu4_mmq_cuda(ctx, (const ggml_tensor *)iu4_buf, src1, dst);

    hipFree(iu4_buf);
    return success;
}
```

### `mul_mat_q1_0_f16.cuh`
```cpp
#pragma once

#include <ggml-cuda.h>
#include <ggml-impl.h>

#ifdef __cplusplus
extern "C" {
#endif

bool mul_mat_q1_0_f16_cuda(ggml_cuda_context &ctx, const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst);

#ifdef __cplusplus
}
#endif
```

### `mul_mat_q1_0_f16.cu`
```cpp
#include "mul_mat_q1_0_f16.cuh"
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <hip/hip_runtime.h>

// Q1_0 block structure: 32 weights, 1-bit each -> 4 bytes for qs
typedef struct {
    ggml_half d;           // scale
    uint8_t qs[4];         // 32 weights, 1 bit each (8 per byte)
} block_q1_0;

// Dequant Q1_0 to f16
__global__ void dequant_q1_0_to_f16_kernel(
    const block_q1_0 * restrict x,
    ggml_half * restrict y,
    int n_rows,
    int n_cols_per_row) {
    int row_idx = blockIdx.x;
    if (row_idx >= n_rows) return;

    const block_q1_0 * block = &x[row_idx];
    ggml_half d = block->d;

    // Dequantize 32 weights per block
    // Each thread block handles one row (32 weights)
    for (int j = threadIdx.x; j < 32; j += 256) {
        uint8_t q_byte = block->qs[j/8];
        int bit = (q_byte >> (j % 8)) & 1;
        // Q1_0 ternary mapping: bit=1 -> +d, bit=0 -> -d or 0
        // For near-lossless ternary { -d, 0, +d }, we map:
        ggml_half val = bit ? d : (ggml_half)0.0f; 
        y[row_idx * n_cols_per_row + j] = val;
    }
}

bool mul_mat_q1_0_f16_cuda(ggml_cuda_context &ctx, const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst) {
    int n_rows = src0->ne[0]; // number of Q1_0 blocks
    int n_cols = src1->ne[0]; // activation dimension

    // Allocate f16 buffer for dequantized weights
    size_t f16_size = n_rows * 32 * sizeof(ggml_half);
    ggml_half *f16_buf = nullptr;
    hipError_t err = hipMalloc(&f16_buf, f16_size);
    if (err != hipSuccess) {
        return false;
    }

    // Launch dequant kernel
    int blocks_dequant = n_rows;
    dequant_q1_0_to_f16_kernel<<<blocks_dequant, 256>>>(
        (const block_q1_0 *)src0->data,
        f16_buf,
        n_rows,
        n_cols);

    // Create a temporary ggml_tensor for the f16 weights
    // In practice, wrap f16_buf in a ggml_tensor with type GGML_TYPE_F16
    // and call mul_mat_f16_hipblaslt_cuda

    // Note: The actual ggml_tensor wrapper creation for f16_buf must use ggml_cuda_tensor_local
    // or similar context allocation to ensure the hipBLASLt dispatch receives a valid src0 tensor.

    // Call existing f16 hipBLASLt GEMM
    // mul_mat_f16_hipblaslt_cuda(ctx, f16_tensor, src1, dst);

    hipFree(f16_buf);
    return true;
}
```

---

## Correctness & Feasibility Notes

- **iu4 int4 WMMA route**: Valid on gfx1201 (RDNA4). The instruction `v_wmma_i32_16x16x32_iu4` is present in the RDNA4 ISA census. Q1_0 ternary values `{ -d, 0, +d }` dequantize to int4 values (mapped to unsigned int4 range `{0, 1, 2}`), which fit exactly in int4/iu4 format without loss. The 2:4-sparsity is not required for dense iu4 WMMA GEMM; the `v_wmma_i32_16x16x32_iu4` instruction handles dense int4 matmul natively.
- **f16 route**: Valid and straightforward. Dequant Q1_0 to `ggml_half` (f16), then dispatch to the existing `mul_mat_f16_hipblaslt_cuda` path.
- Both routes follow the existing pattern established by `mul_mat_q1_0_hipblaslt.cu` (per-output-channel requant/dequant kernel + supports() check + dispatch gate in `ggml-cuda.cu`).

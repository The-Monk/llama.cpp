# Quant-to-IU4-WMMA Route Recipe

Reusable method for porting ANY low-BPW quant type onto the native RDNA4
IU4 WMMA GEMM path (`mul_mat_iu4.cu` / `mul_mat_iu4_mmq.cu`).

---

## 1. Block Layouts (from ggml-common.h)

### block_iu4 (GGML_TYPE_IU4)
```c
struct block_iu4 {
    ggml_half d;                          // per-block scale (fp16)
    uint8_t qs[QK_IU4 / 2];               // packed signed int4, 16 bytes
};
// QK_IU4 = 32
// Packed: byte b, low nibble = element 2b, high nibble = element 2b+1
// Value range: signed int4 = -8 .. +7 (two's complement)
// -1 is encoded as 0xF, -8 as 0x8
```

### block_q1_0 (GGML_TYPE_Q1_0)
```c
struct block_q1_0 {
    ggml_half d;                          // per-block scale (fp16)
    uint8_t qs[QK1_0 / 8];                // 1 bit per element, 16 bytes
};
// QK1_0 = 128
// Binary {-1, +1}: bit set -> +1, clear -> -1
// Bit convention: byte b, low nibble bit e -> element 8b+e,
//                  high nibble bit e -> element 8b+4+e
```

### block_q2_0 (GGML_TYPE_Q2_0, for reference)
```c
struct block_q2_0 {
    ggml_half d;
    uint8_t qs[QK2_0 / 4];                // 2 bits per element, ternary {-1,0,+1}
};
// QK2_0 = 64
```

---

## 2. IU4 WMMA GEMM Entry Point

**File:** `ggml/src/ggml-cuda/mul_mat_iu4.cu`
**Function:** `ggml_cuda_op_mul_mat_iu4(ctx, src0, src1, dst)`
- src0: GGML_TYPE_IU4 weight tensor (N x K)
- src1: GGML_TYPE_F32 activation (K x M)
- dst:  GGML_TYPE_F32 output (N x M)

**Per-block scale application:** Inside `k_mul_mat_iu4`, the accumulator
accumulates `d * sum(w_i * a_i)` where `d = __half2float(blk->d)` and each
`w_i` is a signed int4 value (-8..+7). The scale is applied once per 32-element
block, multiplied into the WMMA accumulator.

**Loader trait interface** (from `mul_mat_iu4_mmq.cu`):
```cpp
struct Loader {
    static __device__ __forceinline__ void load(
        const char *vweight, int64_t n, int64_t nb01, int64_t c,
        int (&w)[4], float &d);
    // Reads ONE 32-element chunk c of weight row n.
    // w[0..3]: 4 packed int32 words (each holds 8 signed int4 values)
    // d:       block scale (float)
};
```

---

## 3. The Requant Map (DERIVED)

### Block ratio
- QK1_0 = 128 elements per Q1_0 block
- QK_IU4 = 32 elements per IU4 block
- **1 Q1_0 block = 4 IU4 blocks** (128 / 32 = 4)

### Scale preservation
- Q1_0 values: ±1 (binary)
- IU4 values: -8..+7 (signed int4)
- Since ±1 fits in signed int4, **no rounding needed** for the weight values
- **Each of the 4 IU4 blocks inherits the SAME d from the parent Q1_0 block**
- Error: **0** (exact reproduction)

### Mapping formula
```
For Q1_0 block index b_q1:
  For each sub-chunk c in 0..3:
    IU4 block index b_iu4 = 4 * b_q1 + c
    IU4 block scale d_iu4 = d_q1_0
    IU4 qs bytes = Q1_0 qs bytes for elements [32*c .. 32*c+31]
      (same bit pattern, interpreted as signed int4 instead of binary)
```

### Generalization to other quants
```
For any quant with block size QK_X and IU4 block size QK_IU4:
  blocks_per_X_block = QK_X / QK_IU4
  For each sub-chunk c in 0..(blocks_per_X_block - 1):
    IU4 block scale d_iu4 = d_X
    IU4 qs = requant(X values in [32*c..32*c+31]) to signed int4 (-8..+7)
```

**Scale-preservation rule:** The per-block scale `d` from the source quant
MUST be propagated unchanged to every IU4 block it covers. The IU4 kernel
multiplies the accumulator by `d` once per 32-element chunk; if you changed
`d` per sub-block you'd introduce a scaling error.

---

## 4. Step-by-Step Porting Procedure

To port quant X onto the IU4 WMMA route:

1. **Verify value range fits signed int4.**
   - Q1_0: ±1 ✓
   - Q2_0: {-1,0,+1} ✓
   - Q4_K: values in [-7,+8] after dequant ✓ (but needs per-4-bit group scale)
   - Q8_0: values in [-128,+127] ✗ (does NOT fit signed int4; would need
     a different route or a different int format)

2. **Compute block ratio.**
   - `ratio = QK_X / QK_IU4` (must be integer)

3. **Write a loader trait** matching the `Loader` interface above.
   - Read source quant blocks
   - Unpack values into `int w[4]` (4 packed int32 words = 32 int4 values)
   - Set `d` from the source block's scale

4. **Write a host-side launcher** that:
   - Allocates a temporary GGML_TYPE_IU4 buffer (N × K)
   - Launches a device kernel to requant source → IU4 (or do it on host)
   - Calls `ggml_cuda_op_mul_mat_iu4(ctx, iu4_tensor, src1, dst)`
   - Frees the temporary buffer

5. **Write a `supports()` function** checking:
   - `src0->type == GGML_TYPE_X`
   - `src1->type == GGML_TYPE_F32`
   - `dst->type == GGML_TYPE_F32`
   - `src0->ne[0] == src1->ne[0]`
   - `src0->ne[0] % QK_IU4 == 0`
   - RDNA4 device check (if applicable)

6. **Add dispatch gate** in `ggml-cuda.cu` near line 2926:
   - Env var gate (opt-in)
   - Call `supports()`, then launcher
   - Fall through on failure

7. **Include the .cuh header** in `ggml-cuda.cu` (around line 89).

---

## 5. Toolchain Notes

- **Do NOT wrap host-side launcher in `#if defined(RDNA4)`.**
  `RDNA4` is defined from `__GFX12__` which clang's HIP frontend only
  predefines during the DEVICE compilation pass. The HOST pass never sees it.
  Keep HOST wrapper unconditional; gate only `__device__/__global__` code.

- **The IU4 path is experimental/model-blocked.** No rotation-free packed-int4
  W4A4 model exists yet (needs QuaRot/SpinQuant or co-trained BitNet-a4.8).
  Plain RTN quantization is a placeholder.

- **Existing Q1_0Loader in mul_mat_iu4_mmq.cu already implements this map.**
  It's the reference implementation for the mmq (multi-warp) variant.
  The new file targets the single-warp-per-tile variant (mul_mat_iu4.cu).

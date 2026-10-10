#include "common.cuh"

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst);

// T434 M3: the signed FWHT (1024 blocks) fused with the per-token int8 quantize of the N1 layout. `y` is the N1TOK act
// buffer (int8 X tiles then fp32 sx[token]); `ntok` tokens of x->ne[0] columns. When write_fp32 is false dst is not
// written. Returns false (nothing launched) for an unsupported shape.
// up != nullptr: the FWHT input is silu(src) * up (a swiglu_split node fused in; src = gate), rows 2D with their own strides.
bool ggml_cuda_op_fwht_signed_quant_n1t(ggml_backend_cuda_context & ctx, const ggml_tensor * src, const ggml_tensor * up,
                                        const ggml_tensor * signs, ggml_tensor * dst, bool write_fp32, void * y,
                                        int perm_hd = 0, int perm_nk = 0, int perm_rep = 0);   // perm_rep > 0: src is the tiled GDN output (see fwht.cu)
// As above, and also writes the Q2_FIELD-layout q8_1 quantization of dst into qy (N=1024 only; false = declined).
bool ggml_cuda_op_fwht_signed_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                                   const ggml_tensor * signs, ggml_tensor * dst, void * qy);
// As above, but src is the TILED [hd, nk, rep] GDN output and the transform input is its grouped [hd, rep, nk] reorder
// (build_lora_mm's reshape/permute/cont folded into the load). Single token only.
bool ggml_cuda_op_fwht_signed_q8_1_perm(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                                        const ggml_tensor * signs, ggml_tensor * dst, void * qy,
                                        int perm_hd, int perm_nk, int perm_rep);

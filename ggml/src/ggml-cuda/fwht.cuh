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

// T492: GDN gated norm (rms_norm(x) * nw * silu(z), 128-wide rows) + the ssm_out tiled->grouped reorder (perm_nk x perm_rep heads of 128)
// + signed FWHT (1024 blocks) + per-token int8 quantize (N1 layout) in one launch; nothing but the int8 bytes and scales is written.
bool ggml_cuda_op_gn_perm_fwht_quant_n1t(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * nw,
                                         const ggml_tensor * z, const ggml_tensor * signs, const ggml_tensor * mm, float eps,
                                         int perm_nk, int perm_rep, void * y, float * dbg = nullptr);   // dbg: verify only, receives the gated-norm values

// T492: attention output gate sigmoid(gview) * cur (gview = strided [hd, heads, ntok] view of the Q+gate projection, read in place) +
// signed FWHT + per-token int8 quantize in one launch (cur and the result are [heads * hd, ntok]); only the int8 bytes are written.
bool ggml_cuda_op_attn_gate_fwht_quant_n1t(ggml_backend_cuda_context & ctx, const ggml_tensor * gview, const ggml_tensor * cur,
                                           const ggml_tensor * signs, const ggml_tensor * mm, void * y);

// T439: residual ADD + RMS_NORM + MUL(weight) + signed FWHT (1024 blocks) + per-token int8 quantize (N1 layout) in one launch.
// sum = the ADD node (xa + xb, always written), norm_dst = the MUL(weight) node value (written only if non-null),
// mm = the FWHT node (written only if write_fp32). y = N1TOK act buffer. false = declined (nothing launched).
// xb == nullptr (T492, 5120-wide rows only): xa is already the sum (written by the GEMM epilogue), nothing is added or stored.
bool ggml_cuda_op_add_norm_fwht_quant_n1t(ggml_backend_cuda_context & ctx, const ggml_tensor * xa, const ggml_tensor * xb,
                                          ggml_tensor * sum, const ggml_tensor * mulw, const ggml_tensor * signs,
                                          ggml_tensor * norm_dst, ggml_tensor * mm, float eps, bool write_fp32, void * y,
                                          void * norm_bf16 = nullptr);   // norm_bf16 != nullptr: the normalized row is stored there as bf16 instead (norm_dst then only names the node)

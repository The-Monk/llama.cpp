#include "common.cuh"

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst);
// As above, and also writes the Q2_FIELD-layout q8_1 quantization of dst into qy (N=1024 only; false = declined).
bool ggml_cuda_op_fwht_signed_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                                   const ggml_tensor * signs, ggml_tensor * dst, void * qy);
// As above, but src is the TILED [hd, nk, rep] GDN output and the transform input is its grouped [hd, rep, nk] reorder
// (build_lora_mm's reshape/permute/cont folded into the load). Single token only.
bool ggml_cuda_op_fwht_signed_q8_1_perm(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                                        const ggml_tensor * signs, ggml_tensor * dst, void * qy,
                                        int perm_hd, int perm_nk, int perm_rep);

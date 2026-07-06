// T80: F8E4M3 KV-cache FA vec-kernel instantiation. Hand-written (not
// generate_cu_files.py, which does not know about F8E4M3 as a KV type) --
// mirrors fattn-vec-instance-q8_0-q8_0.cu.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F8E4M3, GGML_TYPE_F8E4M3);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_F8E4M3, GGML_TYPE_F8E4M3);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_F8E4M3, GGML_TYPE_F8E4M3);

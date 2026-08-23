#pragma once
#include <cstddef>

// Invalidation registry for the hipBLASLt per-route converted-weight caches.
//
// Each mul_mat_*_hipblaslt.cu keeps a cache of weights it has already converted
// (requantised to int8 or e4m3), keyed on the weight tensor's device address.
// That key is only unique while the underlying buffer is alive: once a buffer is
// freed, a later allocation can land on the same address and would silently hit
// a stale entry. That is not hypothetical -- it produces garbage output on
// multi-model runs and on llama-server model swaps.
//
// So every route registers an invalidator here, and the CUDA backend calls
// ggml_hipblaslt_wcache_invalidate() when it frees a device buffer. Cost is
// zero on the hot path; the work happens only at teardown.

typedef void (*ggml_hipblaslt_wcache_invalidator)(const void * base, size_t size);

void ggml_hipblaslt_wcache_register(ggml_hipblaslt_wcache_invalidator fn);

// Every registered function MUST be unregistered before the library that owns it
// is unloaded. ggml_backend_load_all() can find libggml-hip twice (once via
// LD_LIBRARY_PATH and once in the directory of the executable); under RTLD_GLOBAL
// both copies of the registrar resolve to the FIRST copy of the registry, so the
// second copy of the library leaves its function pointers in the first copy of the
// vector. When that second copy is then dlclosed, invalidate() would call into an
// unmapped page and segfault at teardown. Each registrar therefore unregisters
// itself from its destructor, which runs at library fini -- before the unmap.
void ggml_hipblaslt_wcache_unregister(ggml_hipblaslt_wcache_invalidator fn);

// Drop every cached entry whose weight pointer lies in [base, base+size).
void ggml_hipblaslt_wcache_invalidate(const void * base, size_t size);

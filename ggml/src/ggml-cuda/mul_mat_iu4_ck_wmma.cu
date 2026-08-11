// See mul_mat_iu4_ck_wmma.cuh for the full status/doctrine (SCAFFOLDING
// ONLY, gated OFF, not validated -- read that file first).
#include "mul_mat_iu4_ck_wmma.cuh"

// Optional: only compiled against real CK headers if the local ck-ref
// checkout's include dir was added for THIS translation unit (see
// ggml-hip/CMakeLists.txt's GGML_HIP_CK_REF_DIR block). On any machine
// without that local checkout, this falls back to the stub below --
// harmless, since ggml_cuda_iu4_ck_wmma_supports() already unconditionally
// returns false regardless of which branch compiled.
#if __has_include("ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_b_scale.hpp")
#define GGML_IU4_CK_WMMA_HAVE_CK 1
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_b_scale.hpp"
#include "ck/utility/data_type.hpp"
#endif

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

bool ggml_cuda_iu4_ck_wmma_supports(const ggml_tensor * /*src0*/, const ggml_tensor * /*src1*/, const ggml_tensor * /*dst*/) {
    // Unconditionally false -- see the .cuh doctrine comment. Flip only
    // after ggml_cuda_mul_mat_iu4_ck_wmma_selftest() below reports PASS.
    return false;
}

bool ggml_cuda_op_mul_mat_iu4_ck_wmma(ggml_backend_cuda_context & /*ctx*/, const ggml_tensor * /*src0*/, const ggml_tensor * /*src1*/, const ggml_tensor * /*dst*/) {
    GGML_ABORT("mul_mat_iu4_ck_wmma: not reachable -- ggml_cuda_iu4_ck_wmma_supports() always returns false (see .cuh doctrine comment)");
}

#if defined(GGML_IU4_CK_WMMA_HAVE_CK)
namespace ggml_cuda_iu4_ck_wmma_selftest_detail {

using ADataType        = ck::half_t;
using BDataType        = ck::pk_i4_t;
using BScaleDataType   = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using CDataType        = ck::half_t;

using CkRow = ck::tensor_layout::gemm::RowMajor;
using CkCol = ck::tensor_layout::gemm::ColumnMajor;
using ALayout = CkRow;
using BLayout = CkCol;
using CLayout = CkRow;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
// PermuteB left FALSE (the reduced-risk configuration probed this session)
// -- PermuteB=true's host-side reshape could not be validated (see .cuh),
// and PermuteB=false, while it compiles/links/runs, was ALSO measured
// producing wrong results (both nibble-order conventions tried). Kept as
// `false` here so the failure mode this selftest reports matches exactly
// what was measured, rather than silently trying yet another unvalidated
// guess.
static constexpr bool PermuteA = false;
static constexpr bool PermuteB = false;
static constexpr ck::index_t Scale_Block_N = 1;
static constexpr ck::index_t Scale_Block_K = 128;
static constexpr ck::index_t KPerBlock = 64;

using DeviceGemmV2Instance =
    ck::tensor_operation::device::DeviceGemm_BScale_Wmma_CShuffleV3<
        ALayout,   BLayout,  CLayout,
        ADataType, BDataType, BScaleDataType, CDataType, AccDataType, CShuffleDataType,
        PassThrough, PassThrough, PassThrough, GemmDefault,
        256, Scale_Block_N, Scale_Block_K,
        128, 128,
        KPerBlock, 8, 8,
        16,  16,
        4,    2,
        ck::Sequence<8, 32, 1>,  ck::Sequence<1, 0, 2>,  ck::Sequence<1, 0, 2>,
        2, 8, 8, 0,
        ck::Sequence<2, 32, 1>,  ck::Sequence<1, 0, 2>,  ck::Sequence<1, 0, 2>,
        2, 8, 8, 0,
        1, 1, ck::Sequence<1, 32, 1, 8>, 8,
        ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3,
        CDataType, CDataType, PermuteA, PermuteB>;

// Best-effort port of the SIMPLE (PermuteB=false) B-operand packing: plain
// [K,N] column-major (k fastest for fixed n, matching GGML_TYPE_IU4's own
// per-row block layout), 2 int4 values/byte, offset-binary (nib = val+8,
// matching pk_int4_t_to_fp32x2_t's default decode / the b_scale example's
// verification reference `i4 - 8`). Nibble order: reference-decode implies
// k EVEN -> HIGH nibble, k ODD -> LOW nibble (see .cuh doctrine).
static bool run_trial(int M, int N, int K, double & max_abs_err_out, double & max_rel_err_out) {
    std::mt19937 rng(20260720);
    std::uniform_real_distribution<float> actd(-1.0f, 1.0f);
    std::uniform_int_distribution<int> nibd(-8, 7);

    std::vector<ck::half_t> hA((size_t) M * K);
    for (auto & v : hA) {
        v = ck::type_convert<ck::half_t>(actd(rng));
    }

    std::vector<int> logicalB((size_t) K * N);
    for (auto & v : logicalB) {
        v = nibd(rng);
    }
    std::vector<uint8_t> hB((size_t) K * N / 2);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; k += 2) {
            const int hi = logicalB[(size_t) k + (size_t) n * K] + 8;
            const int lo = logicalB[(size_t) (k + 1) + (size_t) n * K] + 8;
            hB[((size_t) k + (size_t) n * K) / 2] = (uint8_t) ((hi << 4) | lo);
        }
    }

    const int scaleK = (K + Scale_Block_K - 1) / Scale_Block_K;
    std::vector<ck::half_t> hScale((size_t) scaleK * N);
    for (auto & v : hScale) {
        v = ck::type_convert<ck::half_t>(0.02f);
    }

    std::vector<double> ref((size_t) M * N, 0.0);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                const float av  = ck::type_convert<float>(hA[(size_t) m * K + k]);
                const int   bv  = logicalB[(size_t) k + (size_t) n * K];
                const float scl = ck::type_convert<float>(hScale[(size_t) (k / Scale_Block_K) * N + n]);
                acc += (double) av * bv * scl;
            }
            ref[(size_t) m * N + n] = acc;
        }
    }

    void * dA = nullptr, * dB = nullptr, * dScale = nullptr, * dC = nullptr;
    if (hipMalloc(&dA, sizeof(ck::half_t) * hA.size()) != hipSuccess ||
        hipMalloc(&dB, hB.size()) != hipSuccess ||
        hipMalloc(&dScale, sizeof(ck::half_t) * hScale.size()) != hipSuccess ||
        hipMalloc(&dC, sizeof(ck::half_t) * (size_t) M * N) != hipSuccess) {
        GGML_LOG_ERROR("%s: hipMalloc failed\n", __func__);
        return false;
    }
    CUDA_CHECK(hipMemcpy(dA, hA.data(), sizeof(ck::half_t) * hA.size(), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(dB, hB.data(), hB.size(), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(dScale, hScale.data(), sizeof(ck::half_t) * hScale.size(), hipMemcpyHostToDevice));

    DeviceGemmV2Instance gemm{};
    auto invoker = gemm.MakeInvoker();
    const ck::index_t scale_stride_bn = scaleK;
    auto argument = gemm.MakeArgument(
            static_cast<ADataType *>(dA), static_cast<BDataType *>(dB), static_cast<CDataType *>(dC),
            M, N, K, K, K, N, scale_stride_bn, static_cast<BScaleDataType *>(dScale),
            1, PassThrough{}, PassThrough{}, PassThrough{});

    if (!gemm.IsSupportedArgument(argument)) {
        GGML_LOG_INFO("%s: M=%d N=%d K=%d: CK IsSupportedArgument() rejected (PermuteB=false not a supported shape/config here)\n",
                       __func__, M, N, K);
        (void) hipFree(dA); (void) hipFree(dB); (void) hipFree(dScale); (void) hipFree(dC);
        max_abs_err_out = -1.0;
        max_rel_err_out = -1.0;
        return false;
    }

    invoker.Run(argument, StreamConfig{nullptr, false, 0});
    const hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        GGML_LOG_ERROR("%s: CK invoker.Run failed: %s\n", __func__, hipGetErrorString(err));
        (void) hipFree(dA); (void) hipFree(dB); (void) hipFree(dScale); (void) hipFree(dC);
        return false;
    }

    std::vector<ck::half_t> hC((size_t) M * N);
    CUDA_CHECK(hipMemcpy(hC.data(), dC, sizeof(ck::half_t) * hC.size(), hipMemcpyDeviceToHost));
    (void) hipFree(dA); (void) hipFree(dB); (void) hipFree(dScale); (void) hipFree(dC);

    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < hC.size(); ++i) {
        const double got = ck::type_convert<float>(hC[i]);
        const double d   = std::fabs(got - ref[i]);
        const double rel = d / (std::fabs(ref[i]) + 1e-6);
        max_abs = std::max(max_abs, d);
        max_rel = std::max(max_rel, rel);
    }
    max_abs_err_out = max_abs;
    max_rel_err_out = max_rel;
    return max_abs < 0.5 && max_rel < 0.05;
}

} // namespace ggml_cuda_iu4_ck_wmma_selftest_detail
#endif // GGML_IU4_CK_WMMA_HAVE_CK

bool ggml_cuda_mul_mat_iu4_ck_wmma_selftest() {
#if defined(GGML_IU4_CK_WMMA_HAVE_CK)
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc)) {
        return true; // vacuous, matches every other IU4 selftest's convention
    }

    using namespace ggml_cuda_iu4_ck_wmma_selftest_detail;
    bool all_pass = true;
    for (int M : {2, 4, 8, 16}) {
        double max_abs = 0.0, max_rel = 0.0;
        const bool ok = run_trial(M, /*N=*/128, /*K=*/512, max_abs, max_rel);
        all_pass &= ok;
        GGML_LOG_INFO("%s: M=%d N=128 K=512 -> %s (max_abs_err=%.6g max_rel_err=%.6g) "
                       "[EXPECTED FAIL this session -- packing not yet reconciled, see mul_mat_iu4_ck_wmma.cuh]\n",
                       __func__, M, ok ? "PASS" : "FAIL", max_abs, max_rel);
    }
    return all_pass;
#else
    GGML_LOG_INFO("%s: CK headers not found at build time (GGML_HIP_CK_REF_DIR) -- nothing to test, vacuously true\n", __func__);
    return true;
#endif
}

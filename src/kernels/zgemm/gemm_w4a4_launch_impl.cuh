#include "gemm_w4a4_launch.cuh"

namespace nunchaku::kernels {

template<typename Config, bool USE_FP4>
void GEMM_W4A4_Launch<Config, USE_FP4>::quantize_w4a4_act_fuse_sparse_mult(Tensor input,
                                                                    Tensor output,
                                                                    Tensor oscales,
                                                                    Tensor lora_down,
                                                                    Tensor lora_act_out,
                                                                    Tensor smooth,
                                                                    bool fuse_glu,
                                                                    bool fp4) {
    const int actualM = input.numel() / input.shape[-1];
    const int actualN = input.shape[-1];

    const int M = ceilDiv(actualM, GEMM::BLOCK_M) * GEMM::BLOCK_M;
    const int N = ceilDiv(actualN / (fuse_glu ? 2 : 1), GEMM::BLOCK_N) * GEMM::BLOCK_N;

    assert(output.dtype() == Tensor::INT8);
    assert(output.numel() / output.shape[-1] == M);
    assert(output.shape[-1] == N / 2);

    // assert(oscales.dtype() == Tensor::FP16);
    if (fp4) {
        assert(oscales.dtype() == Tensor::FP8_E4M3);
        assert(oscales.numel() == M * N / GEMM::WARP_K * 4);
    } else {
        assert(isTypeMatch<half_t>(oscales.dtype()));
        assert(oscales.numel() == M * N / GEMM::WARP_K);
    }

    const int rank = lora_down.shape[1];

    assert(rank % 16 == 0);

    assert(lora_down.shape[0] == N);
    // assert(lora_down.shape[1] == Lora::LORA_RANK);
    assert(lora_act_out.shape[0] == M);
    assert(lora_act_out.shape[1] == rank);

    lora_act_out.zero_();

    dim3 grid(M / GEMM::BLOCK_M, N / GEMM::BLOCK_N);

    // dispatchVal(rank, LoraRanks(), [&]<int RANK>() {
    dispatchBool(fuse_glu, [&]<bool FUSE_GLU>() {
        // using Lora = typename GEMM::Lora<RANK>;
        using kernel = typename GEMM::quantize_w4a4_fuse_sparse_mult_kernel<FUSE_GLU, USE_FP4>;

        auto func = invoke_kernel<kernel, typename kernel::Arguments>;

        checkCUDA(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, kernel::SHMEM_SIZE));

        // log(std::format("quantize_w4a4_act_fuse_lora M={} N={} input={} output={} (size={} numel={})", M, N,
        // input.data_ptr(), output.data_ptr(), output.buffer->getSize(), output.numel()));

        func<<<grid, GEMM::WARP_SIZE * GEMM::NUM_WARPS, kernel::SHMEM_SIZE, getCurrentCUDAStream()>>>(
            typename kernel::Arguments{
                .input         = input.data_ptr<half_t>(),
                .smooth_factor = smooth.valid() ? smooth.data_ptr<packed_wscale_t>() : nullptr,
                .output        = output.data_ptr<packed_act_t>(),
                .oscales       = oscales.data_ptr<typename kernel::oscales_t>(),
                .lora_wgt_down = lora_down.data_ptr<packed_fpsum_t>(),
                .lora_act      = lora_act_out.data_ptr<float>(),
                .lora_rank     = rank,
                .M             = M,
                .N             = N,
                .actualM       = actualM,
                .actualN       = actualN,
                .alwaysfalse   = false,
            });
        checkCUDA(cudaGetLastError());
    });
    // });
}

template<typename Config, bool USE_FP4>
void GEMM_W4A4_Launch<Config, USE_FP4>::quantize_w4a4_act(Tensor input, Tensor output, Tensor oscales) {
    if constexpr (USE_FP4) {
        assert(false); // not implemented
        return;
    }

    int M = input.numel() / input.shape[-1];
    int K = input.shape[-1];

    assert(output.dtype() == Tensor::INT8);
    assert(output.numel() / output.shape[-1] == M);

    // 量化到INT4，但是打包成为INT8，每个结果元素包含两个INT4值，所以输出的最后一维是K/2
    assert(output.shape[-1] == K / 2);

    // assert(oscales.dtype() == Tensor::FP16);
    assert(isTypeMatch<half_t>(oscales.dtype()));
    assert(oscales.numel() == M * K / GEMM::WARP_K);

    dim3 grid(M / GEMM::WARP_M, K / GEMM::WARP_K);
    invoke_kernel<typename GEMM::quantize_w4a4_act_kernel><<<grid, GEMM::WARP_SIZE, 0, getCurrentCUDAStream()>>>(
        input.data_ptr<half_t>(), output.data_ptr<packed_act_t>(), oscales.data_ptr<packed_ascale_t>(), K);
    checkCUDA(cudaGetLastError());
}


}; // namespace nunchaku::kernels

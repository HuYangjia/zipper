#include "kernel.h"
#include "gemm_w4a4_launch.cuh"
// #include "gemm_w4a4_launch_impl.cuh"


namespace nunchaku::kernels {


template<typename F>
static void invoke_launch(Tensor::ScalarType dtype, bool use_fp4, bool fasterI2F, F &&launch) {
    if (fasterI2F && dtype == Tensor::FP16) {
        launch.template operator()<GEMMConfig_W4A4_FP16_FasterI2F, false>();
    } else {
        dispatchBool(use_fp4, [&]<bool USE_FP4>() {
            if (dtype == Tensor::FP16) {
                launch.template operator()<GEMMConfig_W4A4_FP16, USE_FP4>();
            } else if (dtype == Tensor::BF16) {
                launch.template operator()<GEMMConfig_W4A4_BF16, USE_FP4>();
            } else {
                assert(false);
            }
        });
    }
}

void quantize_w4a4_act_fuse_sparse_mult(Tensor input,
                                 Tensor output,
                                 Tensor oscales,
                                 Tensor lora_down,
                                 Tensor lora_act_out,
                                 Tensor smooth,
                                 bool fuse_glu,
                                 bool fp4) {
    invoke_launch(input.dtype(), fp4, false, [&]<typename Config, bool USE_FP4>() {
        GEMM_W4A4_Launch<Config, USE_FP4>::quantize_w4a4_act_fuse_sparse_mult(
            input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4);
    });
}

void quantize_w4a4_act(Tensor input, Tensor output, Tensor oscales) {
    invoke_launch(input.dtype(), false, false, [&]<typename Config, bool USE_FP4>() {
        GEMM_W4A4_Launch<Config, false>::quantize_w4a4_act(input, output, oscales);
    });
}

}; // namespace nunchaku::kernels

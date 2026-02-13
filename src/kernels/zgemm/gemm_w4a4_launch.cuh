#include "gemm_w4a4.cuh"
#include "epilogues.cuh"

namespace nunchaku::kernels {

template<typename Config, bool USE_FP4>
class GEMM_W4A4_Launch {
    using GEMM      = GEMM_W4A4<Config>;
    using Epilogues = Epilogues<Config>;
    using Lora      = Lora<Config>;

    using packed_act_t     = typename GEMM::packed_act_t;
    using packed_wgt_t     = typename GEMM::packed_wgt_t;
    using packed_ascale_t  = typename GEMM::packed_ascale_t;
    using packed_wscale_t  = typename GEMM::packed_wscale_t;
    using packed_amscale_t = typename GEMM::packed_amscale_t;
    using packed_wmscale_t = typename GEMM::packed_wmscale_t;
    using packed_fpsum_t   = typename GEMM::packed_fpsum_t;
    using half_t           = typename GEMM::half_t;

public:
    static void quantize_w4a4_act_fuse_sparse_mult(Tensor input,
                                            Tensor output,
                                            Tensor oscales,
                                            Tensor lora_down,
                                            Tensor lora_act_out,
                                            Tensor smooth,
                                            bool fuse_glu,
                                            bool fp4);
    static void quantize_w4a4_act(Tensor input, Tensor output, Tensor oscales);
    
};

}; // namespace nunchaku::kernels

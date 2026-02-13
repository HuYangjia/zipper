#pragma once

#include "common.h"
#include "Tensor.h"

namespace nunchaku::kernels {

void quantize_w4a4_act_fuse_sparse_mult(Tensor input,
                                 Tensor output,
                                 Tensor oscales,
                                 Tensor lora_down,
                                 Tensor lora_act_out,
                                 Tensor smooth = {},
                                 bool fuse_glu = false,
                                 bool fp4      = false);
void quantize_w4a4_act(Tensor input, Tensor output, Tensor oscales);

void spmm_csr_fp16_wxt(const Tensor& X,
                        const Tensor& csrOffsets,
                        const Tensor& csrCols,
                        const Tensor& csrValsFp16,
                        Tensor& Y);

}; // namespace nunchaku::kernels

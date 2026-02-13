#pragma once

#include "interop/torch.h"
#include "kernels/zgemm/kernel.h"

namespace nunchaku::ops {


void quantize_w4a4_act_fuse_sparse_mult(std::optional<torch::Tensor> input,
                                 std::optional<torch::Tensor> output,
                                 std::optional<torch::Tensor> oscales,
                                 std::optional<torch::Tensor> lora_down,
                                 std::optional<torch::Tensor> lora_act_out,
                                 std::optional<torch::Tensor> smooth,
                                 bool fuse_glu,
                                 bool fp4) {
    TorchOpContext ctx;

    spdlog::trace("running quantize_w4a4_act_fuse_sparse_mult: ");

    auto getTensor = [](std::optional<torch::Tensor> &t) {
        Tensor ret = t.has_value() ? from_torch(t.value()) : Tensor{};
        if (ret.valid()) {
            spdlog::trace("  {}", ret.shape.str());
        } else {
            spdlog::trace("  <invalid>");
        }
        return ret;
    };
    nunchaku::kernels::quantize_w4a4_act_fuse_sparse_mult(getTensor(input),
                                                   getTensor(output),
                                                   getTensor(oscales),
                                                   getTensor(lora_down),
                                                   getTensor(lora_act_out),
                                                   getTensor(smooth),
                                                   fuse_glu,
                                                   fp4);
}

void quantize_w4a4_act(std::optional<torch::Tensor> input,
                         std::optional<torch::Tensor> output,
                         std::optional<torch::Tensor> oscales) {
    TorchOpContext ctx;

    spdlog::trace("running quantize_w4a4_act: ");

    auto getTensor = [](std::optional<torch::Tensor> &t) {
        Tensor ret = t.has_value() ? from_torch(t.value()) : Tensor{};
        if (ret.valid()) {
            spdlog::trace("  {}", ret.shape.str());
        } else {
            spdlog::trace("  <invalid>");
        }
        return ret;
    };
    nunchaku::kernels::quantize_w4a4_act(getTensor(input),
                                 getTensor(output),
                                 getTensor(oscales));
}

void spmm_csr_fp16_wxt(std::optional<torch::Tensor> X_opt,
                       std::optional<torch::Tensor> csrOffsets_opt,
                       std::optional<torch::Tensor> csrCols_opt,
                       std::optional<torch::Tensor> csrValsFp16_opt,
                       std::optional<torch::Tensor> Y_opt) {

    TorchOpContext ctx;

    spdlog::trace("running spmm_csr_fp16_wxt: ");

    auto getTensor = [](std::optional<torch::Tensor> &t) {
        Tensor ret = t.has_value() ? from_torch(t.value()) : Tensor{};
        if (ret.valid()) {
            spdlog::trace("  {}", ret.shape.str());
        } else {
            spdlog::trace("  <invalid>");
        }
        return ret;
    };

    // // 建议：SpMM 这几个输入/输出都必须存在，否则后面很容易崩
    // if (!X_opt.has_value() || !csrOffsets_opt.has_value() || !csrCols_opt.has_value() ||
    //     !csrValsFp16_opt.has_value() || !Y_opt.has_value()) {
    //     throw std::runtime_error("spmm_csr_fp16_wxt: X/csrOffsets/csrCols/csrValsFp16/Y must all be provided");
    // }

    // 关键：先落到具名变量，避免把临时对象绑定到 Tensor&（尤其是 Y）
    Tensor X          = getTensor(X_opt);
    Tensor csrOffsets = getTensor(csrOffsets_opt);
    Tensor csrCols    = getTensor(csrCols_opt);
    Tensor csrValsFp16= getTensor(csrValsFp16_opt);
    Tensor Y          = getTensor(Y_opt);

    // // 再做一次有效性检查更稳（可选）
    // if (!X.valid() || !csrOffsets.valid() || !csrCols.valid() || !csrValsFp16.valid() || !Y.valid()) {
    //     throw std::runtime_error("spmm_csr_fp16_wxt: one of the tensors is invalid after from_torch()");
    // }

    // 传具名变量进去：如果 kernel 端签名是 (const Tensor&, ..., Tensor&)，这里就完全匹配
    nunchaku::kernels::spmm_csr_fp16_wxt(
        X,
        csrOffsets,
        csrCols,
        csrValsFp16,
        Y
    );
}
}
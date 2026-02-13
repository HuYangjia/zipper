#include "ops.h"
#include <torch/extension.h>
#include "interop/torch.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def_submodule("ops")
        // .def("gemm_w4a4", nunchaku::ops::gemm_w4a4)
        // .def("quantize_w4a4_act_fuse_sparse_mult", nunchaku::ops::quantize_w4a4_act_fuse_sparse_mult)
        .def("quantize_w4a4_act", nunchaku::ops::quantize_w4a4_act)
        .def("spmm_csr_fp16_wxt", nunchaku::ops::spmm_csr_fp16_wxt);


    // m.def_submodule("utils")
    //     .def("set_log_level", [](const std::string &level) { spdlog::set_level(spdlog::level::from_str(level)); })
    //     .def("set_cuda_stack_limit", nunchaku::utils::set_cuda_stack_limit)
    //     .def("disable_memory_auto_release", nunchaku::utils::disable_memory_auto_release)
    //     .def("trim_memory", nunchaku::utils::trim_memory)
    //     .def("set_faster_i2f_mode", nunchaku::utils::set_faster_i2f_mode);
}

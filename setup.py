import os
import re
import subprocess
import sys
import setuptools
import torch
from packaging import version as packaging_version
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension


class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        for ext in self.extensions:
            ext.extra_compile_args.setdefault("cxx", [])
            ext.extra_compile_args.setdefault("nvcc", [])
            if self.compiler.compiler_type == "msvc":
                ext.extra_compile_args["cxx"] += ext.extra_compile_args.get("msvc", [])
                ext.extra_compile_args["nvcc"] += ext.extra_compile_args.get("nvcc_msvc", [])
            else:
                ext.extra_compile_args["cxx"] += ext.extra_compile_args.get("gcc", [])
        super().build_extensions()


def get_sm_targets() -> list[str]:
    nvcc_path = os.path.join(CUDA_HOME, "bin/nvcc") if CUDA_HOME else "nvcc"
    nvcc_output = subprocess.check_output([nvcc_path, "--version"]).decode()
    match = re.search(r"release (\d+\.\d+), V(\d+\.\d+\.\d+)", nvcc_output)
    if not match:
        raise RuntimeError("nvcc version not found")
    nvcc_version = match.group(2)

    support_sm120 = packaging_version.parse(nvcc_version) >= packaging_version.parse("12.8")
    support_sm121 = packaging_version.parse(nvcc_version) >= packaging_version.parse("13.0")

    install_mode = os.getenv("NUNCHAKU_INSTALL_MODE", "FAST")
    if install_mode == "FAST":
        ret = []
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            sm = f"{cap[0]}{cap[1]}"
            if sm == "120" and support_sm120:
                sm += "a"
            if sm == "121" and support_sm121:
                sm += "a"
            if sm not in ret:
                ret.append(sm)
    else:
        ret = ["75", "80", "86", "89"]
        if support_sm120:
            ret.append("120a")
        if support_sm121:
            ret.append("121a")
    return ret


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(__file__)

    sm_targets = get_sm_targets()
    assert len(sm_targets) > 0, "No SM targets found"
    print(f"Detected SM targets: {sm_targets}", file=sys.stderr)

    DEBUG = False

    def cond(s):  # debug only
        return [s] if DEBUG else []

    INCLUDE_DIRS = [
        os.path.join(ROOT_DIR, "src"),
        os.path.join(ROOT_DIR, "third_party", "cutlass", "include"),
        os.path.join(ROOT_DIR, "third_party", "spdlog", "include"),
    ]

    GCC_FLAGS = [
        "-DENABLE_BF16=1",
        "-DBUILD_NUNCHAKU=1",
        "-fvisibility=hidden",
        "-g",
        "-std=c++20",
        "-UNDEBUG",
        "-Og",
    ]

    MSVC_FLAGS = [
        "/DENABLE_BF16=1",
        "/DBUILD_NUNCHAKU=1",
        "/std:c++20",
        "/UNDEBUG",
        "/Zc:__cplusplus",
        "/FS",
    ]

    NVCC_FLAGS = [
        "-DENABLE_BF16=1",
        "-DBUILD_NUNCHAKU=1",
        "-g",
        "-std=c++20",
        "-UNDEBUG",
        "-Xcudafe",
        "--diag_suppress=20208",
        *cond("-G"),
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_HALF2_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        f"--threads={len(sm_targets)}",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=--allow-expensive-optimizations=true",
    ]

    # add gencode, and fix compute for 120a/121a
    for target in sm_targets:
        compute = target[:-1] if target.endswith("a") else target
        NVCC_FLAGS += ["-gencode", f"arch=compute_{compute},code=sm_{target}"]

    ext = CUDAExtension(
        name="zipper._C",
        sources=[
            "zipper/csrc/pybind.cpp",
            "src/interop/torch.cpp",
            "src/kernels/zgemm/spmm_int4.cu",
            "src/kernels/zgemm/gemm_w4a4.cu",
            "src/kernels/zgemm/gemm_w4a4_launch_fp16_int4.cu",
            "src/kernels/zgemm/gemm_w4a4_launch_fp16_int4_fasteri2f.cu",
            "src/kernels/zgemm/gemm_w4a4_launch_fp16_fp4.cu",
            "src/kernels/zgemm/gemm_w4a4_launch_bf16_int4.cu",
            "src/kernels/zgemm/gemm_w4a4_launch_bf16_fp4.cu",
            # "src/kernels/zgemm/gemm_w4a4_launch_impl.cuh",
        ],
        include_dirs=INCLUDE_DIRS,
        libraries=["cusparse"],
        extra_compile_args={
            "gcc": GCC_FLAGS,
            "nvcc": NVCC_FLAGS,
            "msvc": MSVC_FLAGS,
            "nvcc_msvc": [],
        },
        # IMPORTANT:
        # Do NOT use "-Wl,--no-undefined" for Python extension modules.
        # Python C-API symbols (PyErr_*, PyGILState_*, PyProperty_Type, ...)
        # are normally resolved by the running Python interpreter at import time.
        extra_link_args=[
            # If cusparse ever gets dropped by the linker on your machine, use:
            # "-Wl,--no-as-needed", "-lcusparse", "-Wl,--as-needed",
        ],
    )

    setuptools.setup(
        name="zipper",
        version="0.0.0",
        packages=setuptools.find_packages(),
        ext_modules=[ext],
        cmdclass={"build_ext": CustomBuildExtension},
    )

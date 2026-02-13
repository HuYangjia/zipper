cmake_minimum_required(VERSION 3.20)
project(spmm_fp16_verify LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# 你的实现（包含 spmm_csr_int4_W_mul_Xt_rowmajorX_fp16 定义）
add_library(spmm_impl STATIC spmm_int4.cu)
target_link_libraries(spmm_impl PRIVATE CUDA::cudart CUDA::cusparse)

# 测试程序
add_executable(spmm_fp16_test spmm_fp16_test.cu)
target_link_libraries(spmm_fp16_test PRIVATE spmm_impl CUDA::cudart CUDA::cusparse)

# 可选：加点调试信息
target_compile_options(spmm_fp16_test PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# 如果你想固定架构（例如 A100=80, 4090=89），可取消注释：
# set_target_properties(spmm_impl PROPERTIES CUDA_ARCHITECTURES "80;86;89")
# set_target_properties(spmm_fp16_test PROPERTIES CUDA_ARCHITECTURES "80;86;89")

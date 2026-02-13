#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define CHECK_CUDA(call) do {                              \
  cudaError_t _e = (call);                                 \
  if (_e != cudaSuccess) return (cusparseStatus_t)CUSPARSE_STATUS_INTERNAL_ERROR; \
} while(0)

#define CHECK_CUSPARSE(call) do {                          \
  cusparseStatus_t _s = (call);                             \
  if (_s != CUSPARSE_STATUS_SUCCESS) return _s;             \
} while(0)

// packed int4: 2 values per byte, low nibble then high nibble
// signed int4 in [-8, 7] -> fp16
__global__ void decode_int4_packed_to_fp16(const uint8_t* __restrict__ in_packed,
                                           __half* __restrict__ out_fp16,
                                           int nnz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nnz) return;

  uint8_t byte = in_packed[i >> 1];
  uint8_t nib  = (i & 1) ? (byte >> 4) : (byte & 0x0F);

  // sign-extend 4-bit to int8
  int8_t v = (int8_t)((int8_t)(nib << 4) >> 4);

  out_fp16[i] = __float2half((float)v);
}

// Computes Y = W * X^T
// W: CSR with INT4 values (packed), shape [M, K], nnz entries
// X: dense FP16, shape [N, K], row-major
// Y: dense FP16, shape [M, N], row-major
cusparseStatus_t spmm_csr_int4_W_mul_Xt_rowmajorX_fp16(
    cusparseHandle_t handle,
    cudaStream_t stream,
    int M, int K, int N,
    int nnz,
    const int* d_csrOffsets,            // [M+1]
    const int* d_columns,               // [nnz]
    const uint8_t* d_values_i4_packed,  // [ceil(nnz/2)]
    const __half* dX_rowmajor_fp16,     // [N, K] row-major
    __half* dY_rowmajor_fp16            // [M, N] row-major
) {
  // 1) Decode W values: int4 -> fp16 (on GPU)
  __half* dW_values_fp16 = nullptr;
  CHECK_CUDA(cudaMalloc(&dW_values_fp16, (size_t)nnz * sizeof(__half)));

  int threads = 256;
  int blocks  = (nnz + threads - 1) / threads;
  decode_int4_packed_to_fp16<<<blocks, threads, 0, stream>>>(d_values_i4_packed, dW_values_fp16, nnz);
  CHECK_CUDA(cudaGetLastError());

  // 2) cuSPARSE descriptors
  CHECK_CUSPARSE(cusparseSetStream(handle, stream));

  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;

  // A: CSR fp16  [M, K]
  CHECK_CUSPARSE(cusparseCreateCsr(
      &matA, M, K, nnz,
      (void*)d_csrOffsets, (void*)d_columns, (void*)dW_values_fp16,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_16F));

  // B: we want X^T with shape [K, N]
  // X is stored row-major as [N, K].
  // Memory(row-major [N,K]) == Memory(column-major [K,N]).
  // So create B as column-major [K, N], ld = K, pointing to dX_rowmajor_fp16.
  CHECK_CUSPARSE(cusparseCreateDnMat(
      &matB, K, N, K,
      (void*)dX_rowmajor_fp16,
      CUDA_R_16F,
      CUSPARSE_ORDER_COL));

  // C: Y is row-major [M, N], ld = N
  CHECK_CUSPARSE(cusparseCreateDnMat(
      &matC, M, N, N,
      (void*)dY_rowmajor_fp16,
      CUDA_R_16F,
      CUSPARSE_ORDER_ROW));

  // 3) SpMM: fp16 inputs, fp32 compute, fp16 output
  float alpha = 1.0f;
  float beta  = 0.0f;

  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(
      handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, matA, matB, &beta, matC,
      CUDA_R_32F,                   // computeType (fp32 accumulate)
      CUSPARSE_SPMM_ALG_DEFAULT,
      &bufferSize));

  void* dBuffer = nullptr;
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

  // Optional preprocess (worth keeping if W sparsity pattern repeats)
  CHECK_CUSPARSE(cusparseSpMM_preprocess(
      handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, matA, matB, &beta, matC,
      CUDA_R_32F,
      CUSPARSE_SPMM_ALG_DEFAULT,
      dBuffer));

  CHECK_CUSPARSE(cusparseSpMM(
      handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, matA, matB, &beta, matC,
      CUDA_R_32F,
      CUSPARSE_SPMM_ALG_DEFAULT,
      dBuffer));

  // 4) cleanup
  CHECK_CUDA(cudaFree(dBuffer));
  CHECK_CUSPARSE(cusparseDestroySpMat(matA));
  CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
  CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
  CHECK_CUDA(cudaFree(dW_values_fp16));

  return CUSPARSE_STATUS_SUCCESS;
}

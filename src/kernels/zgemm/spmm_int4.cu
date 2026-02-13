#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include <stdexcept>
#include <string>

// 你项目里应该已经有 Tensor.h
#include "Tensor.h"

namespace nunchaku::kernels {

static inline void throw_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
    }
}
static inline void throw_cusparse(cusparseStatus_t s, const char* what) {
    if (s != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": " + cusparseGetErrorString(s));
    }
}

static inline void check_tensor_cuda_contig(const Tensor& t, const char* name) {
    if (!t.valid()) throw std::runtime_error(std::string(name) + " is invalid");
    if (!t.is_cuda()) throw std::runtime_error(std::string(name) + " must be CUDA tensor");
    if (!t.is_contiguous()) throw std::runtime_error(std::string(name) + " must be contiguous");
}

static inline void check_dtype(const Tensor& t, Tensor::ScalarType expect, const char* name) {
    if (t.dtype() != expect) {
        throw std::runtime_error(std::string(name) + " dtype mismatch");
    }
}

static inline void check_dim2(const Tensor& t, const char* name) {
    if (t.ndims() != 2) {
        throw std::runtime_error(std::string(name) + " must be 2D");
    }
}
static inline void check_dim1(const Tensor& t, const char* name) {
    if (t.ndims() != 1) {
        throw std::runtime_error(std::string(name) + " must be 1D");
    }
}

// -------------------------------
// Scheme A: one-shot cuSPARSE SpMM
// Y = W * X^T
// W: CSR fp16 [M,K]
// X: fp16 row-major [N,K] (we view it as col-major [K,N])
// Y: fp16 row-major [M,N]
// -------------------------------
static cusparseStatus_t spmm_csr_fp16_w_mul_xt_rowmajorX_fp16_once(
    cusparseHandle_t handle,
    cudaStream_t stream,
    int M, int K, int N,
    int nnz,
    const int* d_csrOffsets,     // [M+1]
    const int* d_columns,        // [nnz]
    const __half* d_values_fp16, // [nnz]
    const __half* dX_rowmajor,   // [N,K] row-major
    __half* dY_rowmajor          // [M,N] row-major
) {
    cusparseStatus_t st;

    st = cusparseSetStream(handle, stream);
    if (st != CUSPARSE_STATUS_SUCCESS) return st;

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnMatDescr_t matB = nullptr;
    cusparseDnMatDescr_t matC = nullptr;

    // A: CSR fp16 [M,K]
    st = cusparseCreateCsr(
        &matA, M, K, nnz,
        (void*)d_csrOffsets, (void*)d_columns, (void*)d_values_fp16,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_16F
    );
    if (st != CUSPARSE_STATUS_SUCCESS) goto CLEANUP;

    // B: treat X row-major [N,K] as col-major [K,N]
    st = cusparseCreateDnMat(
        &matB, K, N, K,
        (void*)dX_rowmajor,
        CUDA_R_16F,
        CUSPARSE_ORDER_COL
    );
    if (st != CUSPARSE_STATUS_SUCCESS) goto CLEANUP;

    // C: Y row-major [M,N]
    st = cusparseCreateDnMat(
        &matC, M, N, N,
        (void*)dY_rowmajor,
        CUDA_R_16F,
        CUSPARSE_ORDER_ROW
    );
    if (st != CUSPARSE_STATUS_SUCCESS) goto CLEANUP;

    {
        float alpha = 1.0f;
        float beta  = 0.0f;

        size_t bufferSize = 0;
        st = cusparseSpMM_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC,
            CUDA_R_32F,                 // fp32 accumulate
            CUSPARSE_SPMM_ALG_DEFAULT,
            &bufferSize
        );
        if (st != CUSPARSE_STATUS_SUCCESS) goto CLEANUP;

        void* dBuffer = nullptr;
        cudaError_t ce = cudaMalloc(&dBuffer, bufferSize);
        if (ce != cudaSuccess) { st = CUSPARSE_STATUS_INTERNAL_ERROR; goto CLEANUP; }

        // preprocess（一次性调用也能跑；你如果后面做 B 版本可把它缓存）
        st = cusparseSpMM_preprocess(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC,
            CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer
        );
        if (st != CUSPARSE_STATUS_SUCCESS) { cudaFree(dBuffer); goto CLEANUP; }

        st = cusparseSpMM(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC,
            CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer
        );

        cudaFree(dBuffer);
        if (st != CUSPARSE_STATUS_SUCCESS) goto CLEANUP;
    }

CLEANUP:
    if (matA) cusparseDestroySpMat(matA);
    if (matB) cusparseDestroyDnMat(matB);
    if (matC) cusparseDestroyDnMat(matC);
    return st;
}

// -------------------------------
// Tensor wrapper (pybind-callable)
// spmm_csr_fp16_wxt(X, offsets, cols, vals, Y)
// -------------------------------
void spmm_csr_fp16_wxt(const Tensor& X,
                             const Tensor& csrOffsets,
                             const Tensor& csrCols,
                             const Tensor& csrValsFp16,
                             Tensor& Y) {
    // basic checks
    check_tensor_cuda_contig(X, "X");
    check_tensor_cuda_contig(csrOffsets, "csrOffsets");
    check_tensor_cuda_contig(csrCols, "csrCols");
    check_tensor_cuda_contig(csrValsFp16, "csrValsFp16");
    check_tensor_cuda_contig(Y, "Y");

    check_dtype(X, Tensor::FP16, "X");
    check_dtype(csrOffsets, Tensor::INT32, "csrOffsets");
    check_dtype(csrCols, Tensor::INT32, "csrCols");
    check_dtype(csrValsFp16, Tensor::FP16, "csrValsFp16");
    check_dtype(Y, Tensor::FP16, "Y");

    check_dim2(X, "X");
    check_dim1(csrOffsets, "csrOffsets");
    check_dim1(csrCols, "csrCols");
    check_dim1(csrValsFp16, "csrValsFp16");
    check_dim2(Y, "Y");

    // layout assumptions: row-major contiguous
    if (X.stride(1) != 1) throw std::runtime_error("X must be row-major contiguous (stride(1)=1)");
    if (Y.stride(1) != 1) throw std::runtime_error("Y must be row-major contiguous (stride(1)=1)");

    int N = X.size(0);
    int K = X.size(1);

    int M = (int)csrOffsets.numel() - 1;
    if (M <= 0) throw std::runtime_error("csrOffsets length must be M+1");

    int nnz = (int)csrValsFp16.numel();
    if ((int)csrCols.numel() != nnz) throw std::runtime_error("csrCols.numel must equal nnz");
    if (Y.size(0) != M || Y.size(1) != N) throw std::runtime_error("Y must have shape [M,N]");

    // set device to match X (assume all tensors on same device)
    throw_cuda(cudaSetDevice(X.get_device()), "cudaSetDevice");

    // stream: 用你项目当前 stream（如果你没有，就先用 0）
    cudaStream_t stream = getCurrentCUDAStream();

    // handle: scheme A 用“每次创建/销毁”（最简单）
    cusparseHandle_t handle = nullptr;
    throw_cusparse(cusparseCreate(&handle), "cusparseCreate");

    auto st = spmm_csr_fp16_w_mul_xt_rowmajorX_fp16_once(
        handle, stream,
        M, K, N, nnz,
        csrOffsets.data_ptr<int>(),
        csrCols.data_ptr<int>(),
        (const __half*)csrValsFp16.data_ptr<void>(),
        (const __half*)X.data_ptr<void>(),
        (__half*)Y.data_ptr<void>());

    cusparseDestroy(handle);

    throw_cusparse(st, "spmm_csr_fp16_w_mul_xt_rowmajorX_fp16_once");
}

} // namespace nunchaku::kernels
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuda_fp16.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#define CUDA_CHECK(call) do {                                      \
  cudaError_t _e = (call);                                         \
  if (_e != cudaSuccess) {                                         \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
            __FILE__, __LINE__, cudaGetErrorString(_e));           \
    exit(1);                                                       \
  }                                                                \
} while(0)

#define CUSPARSE_CHECK(call) do {                                  \
  cusparseStatus_t _s = (call);                                    \
  if (_s != CUSPARSE_STATUS_SUCCESS) {                             \
    fprintf(stderr, "cuSPARSE error %s:%d: %s\n",                  \
            __FILE__, __LINE__, cusparseGetErrorString(_s));       \
    exit(1);                                                       \
  }                                                                \
} while(0)

// 你已有实现的函数：这里用 extern 声明即可（或改成 include 头文件）
extern cusparseStatus_t spmm_csr_int4_W_mul_Xt_rowmajorX_fp16(
    cusparseHandle_t handle,
    cudaStream_t stream,
    int M, int K, int N,
    int nnz,
    const int* d_csrOffsets,
    const int* d_columns,
    const uint8_t* d_values_i4_packed,
    const __half* dX_rowmajor_fp16,
    __half* dY_rowmajor_fp16);

// -------------------------------
// GPU reference: Y1 = W * X^T  (W is CSR int4 packed, X is fp16 row-major)
// output Y1 fp16 row-major [M,N]
// -------------------------------

// 如果你想强制“累加也用 fp16”，把这行改成 1
#ifndef REF_ACCUM_FP16
#define REF_ACCUM_FP16 0
#endif

__device__ __forceinline__ int8_t decode_int4_at(const uint8_t* __restrict__ packed, int idx) {
  uint8_t byte = packed[idx >> 1];
  uint8_t nib  = (idx & 1) ? (byte >> 4) : (byte & 0x0F);
  // sign-extend 4-bit -> int8
  return (int8_t)((int8_t)(nib << 4) >> 4);
}

__global__ void ref_csr_spmm_wxT_fp16(
    int M, int K, int N, int nnz,
    const int* __restrict__ csrOffsets,
    const int* __restrict__ cols,
    const uint8_t* __restrict__ w_i4_packed,
    const __half* __restrict__ X_rowmajor, // [N,K]
    __half* __restrict__ Y_rowmajor        // [M,N]
) {
  int n = blockIdx.x * blockDim.x + threadIdx.x; // column in Y (token index)
  int m = blockIdx.y * blockDim.y + threadIdx.y; // row in Y (out feature)
  if (m >= M || n >= N) return;

  int start = csrOffsets[m];
  int end   = csrOffsets[m + 1];

#if REF_ACCUM_FP16
  __half acc = __float2half(0.0f);
  for (int p = start; p < end; ++p) {
    int k = cols[p];
    int8_t wq = decode_int4_at(w_i4_packed, p);
    __half w = __float2half((float)wq);
    __half x = X_rowmajor[n * K + k];
    acc = __hadd(acc, __hmul(w, x));
  }
  Y_rowmajor[m * N + n] = acc;
#else
  float acc = 0.0f;
  for (int p = start; p < end; ++p) {
    int k = cols[p];
    int8_t wq = decode_int4_at(w_i4_packed, p);
    float w = (float)wq;
    float x = __half2float(X_rowmajor[n * K + k]);
    acc += w * x;
  }
  Y_rowmajor[m * N + n] = __float2half(acc);
#endif
}

// -------------------------------
// helpers: args / random gen / CSR pack / dump / mse
// -------------------------------
static int get_int_arg(int argc, char** argv, const char* key, int defv) {
  for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == key) return atoi(argv[i + 1]);
  return defv;
}
static float get_float_arg(int argc, char** argv, const char* key, float defv) {
  for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == key) return (float)atof(argv[i + 1]);
  return defv;
}
static std::string get_str_arg(int argc, char** argv, const char* key, const char* defv) {
  for (int i = 1; i + 1 < argc; ++i) if (std::string(argv[i]) == key) return std::string(argv[i + 1]);
  return std::string(defv);
}

static void pack_int4_signed_to_u8(const std::vector<int8_t>& v_i4, std::vector<uint8_t>& packed) {
  int nnz = (int)v_i4.size();
  packed.assign((nnz + 1) / 2, 0);

  auto to_nibble = [](int8_t v) -> uint8_t {
    // two's complement keep low 4 bits
    return (uint8_t)(v & 0x0F);
  };

  for (int i = 0; i < nnz; ++i) {
    uint8_t nib = to_nibble(v_i4[i]);
    int bi = i >> 1;
    if ((i & 1) == 0) packed[bi] = (packed[bi] & 0xF0) | nib;
    else              packed[bi] = (packed[bi] & 0x0F) | (uint8_t)(nib << 4);
  }
}

static void build_random_sparse_W_csr_int4packed(
    int M, int K, float density,
    std::vector<int>& offsets,
    std::vector<int>& cols,
    std::vector<int8_t>& vals_i4,
    std::vector<uint8_t>& vals_i4_packed,
    std::vector<__half>& W_dense_fp16 // for dumping [M,K]
) {
  std::mt19937 rng(123);
  std::uniform_int_distribution<int> val_dist(-8, 7);

  int nnz_per_row = std::max(1, (int)std::lround(density * K));

  offsets.resize(M + 1);
  cols.clear();
  vals_i4.clear();
  W_dense_fp16.assign((size_t)M * K, __float2half(0.0f));

  offsets[0] = 0;
  for (int r = 0; r < M; ++r) {
    std::unordered_set<int> used;
    std::vector<int> rowcols;
    rowcols.reserve(nnz_per_row);

    while ((int)rowcols.size() < nnz_per_row) {
      int c = (int)(rng() % (uint32_t)K);
      if (used.insert(c).second) rowcols.push_back(c);
    }
    std::sort(rowcols.begin(), rowcols.end());

    for (int c : rowcols) {
      int v = val_dist(rng);
      if (v == 0) v = 1; // 尽量避免 0 影响 nnz
      if (v < -8) v = -8;
      if (v >  7) v =  7;

      cols.push_back(c);
      vals_i4.push_back((int8_t)v);
      W_dense_fp16[(size_t)r * K + c] = __float2half((float)v);
    }
    offsets[r + 1] = (int)cols.size();
  }

  pack_int4_signed_to_u8(vals_i4, vals_i4_packed);
}

static double mse_fp16(const std::vector<__half>& a, const std::vector<__half>& b) {
  if (a.size() != b.size()) return 1e300;
  long double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    long double da = (long double)__half2float(a[i]);
    long double db = (long double)__half2float(b[i]);
    long double d = da - db;
    s += d * d;
  }
  return (double)(s / (long double)a.size());
}

static void dump_mat_fp16(std::ofstream& ofs, const char* name,
                          const std::vector<__half>& a, int rows, int cols) {
  ofs << name << " (" << rows << "x" << cols << ", fp16 printed as fp32)\n";
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      ofs << __half2float(a[(size_t)i * cols + j]);
      if (j + 1 < cols) ofs << ",";
    }
    ofs << "\n";
  }
  ofs << "\n";
}

static void dump_mat_fp16_ptr(std::ofstream& ofs, const char* name,
                              const __half* a, int rows, int cols) {
  ofs << name << " (" << rows << "x" << cols << ", fp16 printed as fp32)\n";
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      ofs << __half2float(a[(size_t)i * cols + j]);
      if (j + 1 < cols) ofs << ",";
    }
    ofs << "\n";
  }
  ofs << "\n";
}

int main(int argc, char** argv) {
  int M = get_int_arg(argc, argv, "--M", 8);
  int N = get_int_arg(argc, argv, "--N", 6);
  int K = get_int_arg(argc, argv, "--K", 16);
  float density = get_float_arg(argc, argv, "--density", 0.10f);
  std::string out_path = get_str_arg(argc, argv, "--out", "spmm_fp16_dump.txt");

  std::cout << "M=" << M << " N=" << N << " K=" << K
            << " density=" << density << " out=" << out_path << "\n";

  // 1) Random X fp16, shape [N,K] row-major
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> xdist(-2.0f, 2.0f);

  std::vector<__half> hX((size_t)N * K);
  for (int i = 0; i < N * K; ++i) {
    hX[i] = __float2half(xdist(rng));
  }

  // 2) Random sparse W int4 -> CSR + packed
  std::vector<int> hOff, hCol;
  std::vector<int8_t> hVal_i4;
  std::vector<uint8_t> hVal_i4_packed;
  std::vector<__half> hW_dense_fp16;

  build_random_sparse_W_csr_int4packed(M, K, density, hOff, hCol, hVal_i4, hVal_i4_packed, hW_dense_fp16);
  int nnz = (int)hVal_i4.size();
  std::cout << "nnz=" << nnz << " packed_bytes=" << hVal_i4_packed.size() << "\n";

  // device alloc
  int *dOff = nullptr, *dCol = nullptr;
  uint8_t* dVal4 = nullptr;
  __half *dX = nullptr, *dY1 = nullptr, *dY2 = nullptr;

  CUDA_CHECK(cudaMalloc(&dOff, (size_t)(M + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dCol, (size_t)nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dVal4, (size_t)hVal_i4_packed.size() * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&dX, (size_t)N * K * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dY1, (size_t)M * N * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dY2, (size_t)M * N * sizeof(__half)));

  CUDA_CHECK(cudaMemcpy(dOff, hOff.data(), (size_t)(M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dCol, hCol.data(), (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dVal4, hVal_i4_packed.data(), (size_t)hVal_i4_packed.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dX, hX.data(), (size_t)N * K * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dY1, 0, (size_t)M * N * sizeof(__half)));
  CUDA_CHECK(cudaMemset(dY2, 0, (size_t)M * N * sizeof(__half)));

  // create handle/stream
  cusparseHandle_t handle;
  CUSPARSE_CHECK(cusparseCreate(&handle));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // 3) Y1 = W * X^T  reference (GPU kernel, fp16 output)
  dim3 block(16, 8);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  ref_csr_spmm_wxT_fp16<<<grid, block, 0, stream>>>(M, K, N, nnz, dOff, dCol, dVal4, dX, dY1);
  CUDA_CHECK(cudaGetLastError());

  // 4) Y2 = cuSPARSE(W_csr_fp16decoded, X_fp16)
  CUSPARSE_CHECK(spmm_csr_int4_W_mul_Xt_rowmajorX_fp16(
      handle, stream, M, K, N, nnz, dOff, dCol, dVal4, dX, dY2));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // copy back
  std::vector<__half> hY1((size_t)M * N);
  std::vector<__half> hY2((size_t)M * N);
  CUDA_CHECK(cudaMemcpy(hY1.data(), dY1, (size_t)M * N * sizeof(__half), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hY2.data(), dY2, (size_t)M * N * sizeof(__half), cudaMemcpyDeviceToHost));

  double mse = mse_fp16(hY1, hY2);
  std::cout << "MSE(Y1,Y2) = " << mse << "\n";

  // 5) Compute XW^T result (shape [N,M]) in fp16: just transpose of WX^T
  std::vector<__half> hXW_T((size_t)N * M);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      hXW_T[(size_t)n * M + m] = hY1[(size_t)m * N + n];
    }
  }

  // dump all to file
  std::ofstream ofs(out_path);
  ofs << "M=" << M << " N=" << N << " K=" << K << " density=" << density << " nnz=" << nnz << "\n";
#if REF_ACCUM_FP16
  ofs << "REF_ACCUM=fp16\n\n";
#else
  ofs << "REF_ACCUM=fp32_then_cast_fp16\n\n";
#endif

  dump_mat_fp16(ofs, "X_fp16 [N,K] row-major", hX, N, K);
  dump_mat_fp16(ofs, "W_dense_fp16 (from CSR int4) [M,K]", hW_dense_fp16, M, K);
  dump_mat_fp16(ofs, "Y1_ref_fp16 = W * X^T [M,N]", hY1, M, N);
  dump_mat_fp16(ofs, "Y2_cusparse_fp16 [M,N]", hY2, M, N);
  dump_mat_fp16(ofs, "XW^T_fp16 [N,M]", hXW_T, N, M);

  ofs << "MSE_fp16(Y1,Y2)=" << mse << "\n";
  ofs.close();

  // cleanup
  CUSPARSE_CHECK(cusparseDestroy(handle));
  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaFree(dOff));
  CUDA_CHECK(cudaFree(dCol));
  CUDA_CHECK(cudaFree(dVal4));
  CUDA_CHECK(cudaFree(dX));
  CUDA_CHECK(cudaFree(dY1));
  CUDA_CHECK(cudaFree(dY2));

  return 0;
}

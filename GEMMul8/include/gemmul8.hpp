#pragma once
#include "../src/cuda_impl.hpp"
#include <vector>

#define BATCHED

namespace gemmul8 {

// workSize returns work size required in gemm
// Usage:
//  void *work;
//  cudaMalloc(&work, workSize(m,n,k,num_moduli));
size_t workSize(const size_t m,             // Number of rows of C
                const size_t n,             // Number of columns of C
                const size_t k,             // Inner dimension <= 2^17
                const unsigned num_moduli); // #moduli, 2 <= num_moduli <= (DGEMM emulation) ? 20 : 19

// gemm returns computation time in second of each part
// Usage:
//  std::vector<double> times = gemmul8::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, fastmode, work);
//  or
//  gemmul8::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, devA, m, devB, k, &beta, devC, m, num_moduli, fastmode, work);
template <typename T>
std::vector<double> gemm(cublasHandle_t handle,        // Handle to the cuBLAS library context
                         const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
                         const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
                         const size_t m,               // Number of rows of C
                         const size_t n,               // Number of columns of C
                         const size_t k,               // Inner dimension <= 2^17
                         const T *alpha,               // Scaling factor for op(A)*op(B)
                         const T *const A,             // 1-D device array of dimensions lda*k (CUBLAS_OP_N) or lda*m (CUBLAS_OP_T)
                         const size_t lda,             // Leading dimension of A
                         const T *const B,             // 1-D device array of dimensions ldb*n (CUBLAS_OP_N) or ldb*k (CUBLAS_OP_T)
                         const size_t ldb,             // Leading dimension of B
                         const T *beta,                // Scaling factor for C
                         T *const C,                   // 1-D device array of dimensions ldc*n
                         const size_t ldc,             // Leading dimension of C
                         const unsigned num_moduli,    // #moduli, 2 <= num_moduli <= 20
                         const bool fastmode,          // false (accurate mode) or true (fast mode)
                         void *const work);            // workspace allocated in advance

template <>
std::vector<double> gemm<double>(cublasHandle_t handle,        // Handle to the cuBLAS library context
                                 const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
                                 const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
                                 const size_t m,               // Number of rows of C
                                 const size_t n,               // Number of columns of C
                                 const size_t k,               // Inner dimension <= 2^17
                                 const double *alpha,          // Scaling factor for op(A)*op(B)
                                 const double *const A,        // 1-D device array of dimensions lda*k (CUBLAS_OP_N) or lda*m (CUBLAS_OP_T)
                                 const size_t lda,             // Leading dimension of A
                                 const double *const B,        // 1-D device array of dimensions ldb*n (CUBLAS_OP_N) or ldb*k (CUBLAS_OP_T)
                                 const size_t ldb,             // Leading dimension of B
                                 const double *beta,           // Scaling factor for C
                                 double *const C,              // 1-D device array of dimensions ldc*n
                                 const size_t ldc,             // Leading dimension of C
                                 const unsigned num_moduli,    // #moduli, 2 <= num_moduli <= 20
                                 const bool fastmode,          // false (accurate mode) or true (fast mode)
                                 void *const work);            // workspace allocated in advance

template <>
std::vector<double> gemm<float>(cublasHandle_t handle,        // Handle to the cuBLAS library context
                                const cublasOperation_t op_A, // CUBLAS_OP_N or CUBLAS_OP_T
                                const cublasOperation_t op_B, // CUBLAS_OP_N or CUBLAS_OP_T
                                const size_t m,               // Number of rows of C
                                const size_t n,               // Number of columns of C
                                const size_t k,               // Inner dimension <= 2^17
                                const float *alpha,           // Scaling factor for op(A)*op(B)
                                const float *const A,         // 1-D device array of dimensions lda*k (CUBLAS_OP_N) or lda*m (CUBLAS_OP_T)
                                const size_t lda,             // Leading dimension of A
                                const float *const B,         // 1-D device array of dimensions ldb*n (CUBLAS_OP_N) or ldb*k (CUBLAS_OP_T)
                                const size_t ldb,             // Leading dimension of B
                                const float *beta,            // Scaling factor for C
                                float *const C,               // 1-D device array of dimensions ldc*n
                                const size_t ldc,             // Leading dimension of C
                                const unsigned num_moduli,    // #moduli, 2 <= num_moduli <= 19
                                const bool fastmode,          // false (accurate mode) or true (fast mode)
                                void *const work);            // workspace allocated in advance

} // namespace gemmul8

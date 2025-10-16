#include "../include/gemmul8.hpp"
#include "common.hpp"
#include "conv_32i_2_8u.hpp"
#include "inverse_scaling.hpp"
#include "scaling.hpp"
#include "table.hpp"

namespace {
void timing_start(std::chrono::system_clock::time_point &timetmp) {
    cudaDeviceSynchronize();
    timetmp = std::chrono::system_clock::now();
}

void timing_stop(std::chrono::system_clock::time_point &timetmp, double &timer) {
    cudaDeviceSynchronize();
    timer += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - timetmp).count();
}
} // namespace

namespace gemmul8 {

//------------------------------
// Calculating required work size
//------------------------------
size_t workSize(const size_t m,            // Number of rows of C
                const size_t n,            // Number of columns of C
                const size_t k,            // Inner dimension <= 2^17
                const unsigned num_moduli) // #moduli, 2 <= num_moduli <= (DGEMM emulation) ? 20 : 18
{
    const size_t lda8i     = ((k + 15) >> 4) << 4; // multiple of 16
    const size_t ldb8i     = lda8i;
    const size_t m_pad     = ((m + 15) >> 4) << 4; // multiple of 16
    const size_t sizeA     = lda8i * m_pad;
    const size_t sizeB     = ldb8i * n;
    const size_t sizeC     = m_pad * n;
    const size_t size_vecA = m_pad;
    const size_t size_vecB = (((n + 15) >> 4) << 4); // multiple of 16

    size_t total_size = 0;
    total_size += sizeof(int8_t) * (sizeA + sizeB) * num_moduli;
    total_size += sizeof(uint8_t) * sizeC * num_moduli;
#if defined(BATCHED)
    total_size += sizeof(int32_t) * sizeC * num_moduli;
#else
    total_size += sizeof(int32_t) * sizeC;
#endif
    total_size += sizeof(int16_t) * (size_vecA + size_vecB);

    return total_size;
}

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
                                 void *const work)             // workspace allocated in advance
{
    //------------------------------
    // timer
    //------------------------------
    std::chrono::system_clock::time_point timetmp;
    std::vector<double> timer(4, 0.0);

    //------------------------------
    // set constants
    //------------------------------
    const size_t lda8i       = ((k + 15) >> 4) << 4; // multiple of 16
    const size_t ldb8i       = lda8i;
    const size_t m_pad       = ((m + 15) >> 4) << 4; // multiple of 16
    const size_t sizeA       = lda8i * m_pad;
    const size_t sizeB       = ldb8i * n;
    const size_t sizeC       = m_pad * n;
    const size_t size_vecA   = m_pad;
    const unsigned table_idx = num_moduli - 2;
    const unsigned numM      = oz2_table::numM[table_idx]; // numM <= 2
    const bool is_numM_1     = numM == 1;
    constexpr int32_t one    = 1;
    constexpr int32_t zero   = 0;

#if defined(THREADS1)
    oz2_const::threads_scaling = THREADS1;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 80
    oz2_const::threads_scaling = 256;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 89
    oz2_const::threads_scaling = 256;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 90
    oz2_const::threads_scaling = (fastmode) ? 256 : ((lda8i > 4096) ? 512 : 256);
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 120
    oz2_const::threads_scaling = (fastmode) ? 128 : 256;
#else
    oz2_const::threads_scaling = 256;
#endif

#if defined(THREADS2)
    oz2_const::threads_conv32i8u = THREADS2;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 80
    oz2_const::threads_conv32i8u = 128;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 89
    oz2_const::threads_conv32i8u = 128;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 90
    oz2_const::threads_conv32i8u = (sizeC > 16777216) ? 128 : 256;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 120
    oz2_const::threads_conv32i8u = 256;
#else
    oz2_const::threads_conv32i8u = 256;
#endif

#if defined(THREADS3)
    oz2_const::threads_invscaling = THREADS3;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 80
    oz2_const::threads_invscaling = 128;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 89
    oz2_const::threads_invscaling = 128;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 90
    oz2_const::threads_invscaling = 128;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 120
    oz2_const::threads_invscaling = 512;
#else
    oz2_const::threads_invscaling = 256;
#endif

    oz2_const::grids_invscaling = (m * n + oz2_const::threads_invscaling - 1) / oz2_const::threads_invscaling;
    oz2_const::grids_conv32i8u  = ((sizeC >> 2) + oz2_const::threads_conv32i8u - 1) / oz2_const::threads_conv32i8u;

    //------------------------------
    // set workspace (16byte align)
    //------------------------------
    int8_t *A8i   = reinterpret_cast<int8_t *>(work);                       // lda8i*m*sizeod(int8_t)*num_moduli
    int8_t *B8i   = A8i + sizeA * num_moduli;                               // ldb8i*n*sizeod(int8_t)*num_moduli
    uint8_t *C8u  = reinterpret_cast<uint8_t *>(B8i + sizeB * num_moduli);  // (m*n+15)/16*16*sizeof(uint8_t)*num_moduli
    int32_t *C32i = reinterpret_cast<int32_t *>(C8u + sizeC * num_moduli);  // (m*n+15)/16*16*sizeof(int32_t)
    int16_t *sftA
#if defined(BATCHED)
                  = reinterpret_cast<int16_t *>(C32i + sizeC * num_moduli); // (m+15)/16*16*sizeof(int16_t)*num_moduli
#else
                  = reinterpret_cast<int16_t *>(C32i + sizeC);              // (m+15)/16*16*sizeof(int16_t)
#endif
    int16_t *sftB = sftA + size_vecA;                                       // (n+15)/16*16*sizeof(int16_t)

    if (is_numM_1) {
        cudaMemcpyToSymbol(oz2_table::NMi_dev, &oz2_table::NMi_1[table_idx][0], num_moduli * sizeof(double));
    } else {
        cudaMemcpyToSymbol(oz2_table::NMi_dev, &oz2_table::NMi_2[num_moduli - 8][0][0], 2 * num_moduli * sizeof(double));
    }
    cudaMemcpyToSymbol(oz2_table::moduli_dev, oz2_table::moduli, num_moduli * sizeof(oz2_table::tab_t<double>));
#if defined(BATCHED)
    cudaMemcpyToSymbol(oz2_table::invm_32i_dev, oz2_table::invm_32i, (num_moduli - 1) * sizeof(int32_t));
#endif

    //------------------------------
    // Scaling
    // A =: diag(2^sftA) * A64f, A64f is integer
    // B =: B64f * diag(2^sftB), B64f is integer
    // Then, calculating mod for all moduli
    // A8i := A64f - round(A64f/modulus[i])*modulus[i] (-128 <= A8i <= 127)
    // B8i := B64f - round(B64f/modulus[i])*modulus[i] (-128 <= A8i <= 127)
    //------------------------------
    timing_start(timetmp);
    if (fastmode) {
        oz2_util::vecnorm::scaling<double>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, lda8i * m_pad, sftA, B8i, ldb8i, ldb8i * n, sftB, table_idx);
    } else {
        oz2_util::int8tc::scaling<double>(handle, op_A, op_B, m, m_pad, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, lda8i * m_pad, sftA, B8i, ldb8i, ldb8i * n, sftB, C32i, table_idx);
    }
    timing_stop(timetmp, timer[0]);

#if defined(BATCHED)
    timing_start(timetmp);
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m_pad, n, ldb8i, &one, A8i, CUDA_R_8I, ldb8i, sizeA, B8i, CUDA_R_8I, ldb8i, sizeB, &zero, C32i, CUDA_R_32I, m_pad, sizeC, num_moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    timing_stop(timetmp, timer[1]);

    timing_start(timetmp);
    oz2_util::conv_32i_2_8u_batched(num_moduli, sizeC, C32i, C8u);
    timing_stop(timetmp, timer[2]);
#else
    for (unsigned i = 0; i < num_moduli; ++i) {
        //-----------------------------
        // Error-free matrix multiplication
        // C32i := A8i*B8i
        //------------------------------
        timing_start(timetmp);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m_pad, n, lda8i, &one, A8i + i * sizeA, CUDA_R_8I, lda8i, B8i + i * sizeB, CUDA_R_8I, ldb8i, &zero, C32i, CUDA_R_32I, m_pad, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
        timing_stop(timetmp, timer[1]);

        //------------------------------
        // Calculating mod
        // C8u[i] := mod(C32i, modulus[i]) >= 0
        //------------------------------
        timing_start(timetmp);
        oz2_util::conv_32i_2_8u(i, sizeC, C32i, C8u + i * sizeC);
        timing_stop(timetmp, timer[2]);
    }
#endif

    //------------------------------
    // Accumulation and Inverse scaling
    // C64f = sum(Ni*Mi*C8u[i]),
    //  where
    //      Mi := M/modulus[i],
    //      M := prod(modulus[all]),
    //      mod(Ni*Mi, modulus[i]) == 1.
    // C := C64f - round(C64f/M)*M
    // C := diag(2^-sftA) * C * diag(2^-sftB)
    //------------------------------
    timing_start(timetmp);
    oz2_util::inverse_scaling(is_numM_1, num_moduli, m, n, C8u, m_pad, sizeC, C, ldc, sftA, sftB, *alpha, *beta);
    timing_stop(timetmp, timer[3]);

    return timer;
}

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
                                const unsigned num_moduli,    // #moduli, 2 <= num_moduli <= 18
                                const bool fastmode,          // false (accurate mode) or true (fast mode)
                                void *const work)             // workspace allocated in advance
{
    //------------------------------
    // timer
    //------------------------------
    std::chrono::system_clock::time_point timetmp;
    std::vector<double> timer(4, 0.0);

    //------------------------------
    // set constants
    //------------------------------
    const size_t lda8i       = ((k + 15) >> 4) << 4; // multiple of 16
    const size_t ldb8i       = lda8i;
    const size_t m_pad       = ((m + 15) >> 4) << 4; // multiple of 16
    const size_t sizeA       = lda8i * m_pad;
    const size_t sizeB       = ldb8i * n;
    const size_t sizeC       = m_pad * n;
    const size_t size_vecA   = m_pad;
    const unsigned table_idx = num_moduli - 2;
    constexpr int32_t one    = 1;
    constexpr int32_t zero   = 0;

#if defined(THREADS1)
    oz2_const::threads_scaling = THREADS1;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 80
    oz2_const::threads_scaling = 256;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 89
    oz2_const::threads_scaling = (fastmode) ? ((lda8i > 4096) ? 512 : 128) : 128;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 90
    oz2_const::threads_scaling = (fastmode) ? ((lda8i >= 4096) ? 1024 : 256) : 256;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 120
    oz2_const::threads_scaling = (fastmode) ? ((lda8i >= 4096) ? 512 : 256) : 256;
#else
    oz2_const::threads_scaling = 256;
#endif

#if defined(THREADS2)
    oz2_const::threads_conv32i8u = THREADS2;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 80
    oz2_const::threads_conv32i8u = 128;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 89
    oz2_const::threads_conv32i8u = 128;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 90
    oz2_const::threads_conv32i8u = (sizeC > 16777216) ? 128 : 256;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 120
    oz2_const::threads_conv32i8u = 256;
#else
    oz2_const::threads_conv32i8u = 256;
#endif

#if defined(THREADS3)
    oz2_const::threads_invscaling = THREADS3;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 80
    oz2_const::threads_invscaling = 128;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 89
    oz2_const::threads_invscaling = 128;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 90
    oz2_const::threads_invscaling = 128;
#elif defined(GEMMul8_ARCH) && GEMMul8_ARCH == 120
    oz2_const::threads_invscaling = 512;
#else
    oz2_const::threads_invscaling = 256;
#endif

    oz2_const::grids_invscaling = (m * n + oz2_const::threads_invscaling - 1) / oz2_const::threads_invscaling;
    oz2_const::grids_conv32i8u  = ((sizeC >> 2) + oz2_const::threads_conv32i8u - 1) / oz2_const::threads_conv32i8u;

    //------------------------------
    // set workspace (16byte align)
    //------------------------------
    int8_t *A8i   = reinterpret_cast<int8_t *>(work);                      // lda8i*m*sizeod(int8_t)*num_moduli
    int8_t *B8i   = A8i + sizeA * num_moduli;                              // ldb8i*n*sizeod(int8_t)*num_moduli
    uint8_t *C8u  = reinterpret_cast<uint8_t *>(B8i + sizeB * num_moduli); // (m*n+15)/16*16*sizeof(uint8_t)*num_moduli
    int32_t *C32i = reinterpret_cast<int32_t *>(C8u + sizeC * num_moduli); // (m*n+15)/16*16*sizeof(int32_t)
    int16_t *sftA = reinterpret_cast<int16_t *>(C32i + sizeC);             // (m+15)/16*16*sizeof(int16_t)
    int16_t *sftB = sftA + size_vecA;                                      // (n+15)/16*16*sizeof(int16_t)

    cudaMemcpyToSymbol(oz2_table::NMi_dev, &oz2_table::NMi_1[table_idx][0], num_moduli * sizeof(double));
    cudaMemcpyToSymbol(oz2_table::modulif_dev, oz2_table::modulif, num_moduli * sizeof(oz2_table::tab_t<float>));

    //------------------------------
    // Scaling
    // A =: diag(2^sftA) * A32f, A32f is integer
    // B =: B32f * diag(2^sftB), B32f is integer
    // Then, calculating mod for all moduli
    // A8i := mod(A32f, modulus[i]) - 128 (-128 <= A8i <= 127)
    // B8i := mod(B32f, modulus[i]) - 128 (-128 <= A8i <= 127)
    //------------------------------
    timing_start(timetmp);
    if (fastmode) {
        oz2_util::vecnorm::scaling<float>(op_A, op_B, m, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, lda8i * m_pad, sftA, B8i, ldb8i, ldb8i * n, sftB, table_idx);
    } else {
        oz2_util::int8tc::scaling<float>(handle, op_A, op_B, m, m_pad, n, k, num_moduli, A, lda, B, ldb, A8i, lda8i, lda8i * m_pad, sftA, B8i, ldb8i, ldb8i * n, sftB, C32i, table_idx);
    }
    timing_stop(timetmp, timer[0]);

    for (unsigned i = 0; i < num_moduli; ++i) {
        //-----------------------------
        // Error-free matrix multiplication
        // C32i := A8i*B8i
        //------------------------------
        timing_start(timetmp);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m_pad, n, lda8i, &one, A8i + i * sizeA, CUDA_R_8I, lda8i, B8i + i * sizeB, CUDA_R_8I, ldb8i, &zero, C32i, CUDA_R_32I, m_pad, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
        timing_stop(timetmp, timer[1]);

        //------------------------------
        // Calculating mod
        // C8u[i] := mod(C32i, modulus[i]) >= 0
        //------------------------------
        timing_start(timetmp);
        oz2_util::conv_32i_2_8u(i, sizeC, C32i, C8u + i * sizeC);
        timing_stop(timetmp, timer[2]);
    }

    //------------------------------
    // Accumulation and Inverse scaling
    // C32f = sum(Ni*Mi*C8u[i]),
    //  where
    //      Mi := M/modulus[i],
    //      M := prod(modulus[all]),
    //      mod(Ni*Mi, modulus[i]) == 1.
    // C := C32f - round(C32f/M)*M
    // C := diag(2^-sftA) * C * diag(2^-sftB)
    //------------------------------
    timing_start(timetmp);
    oz2_util::inverse_scaling(num_moduli, m, n, C8u, m_pad, sizeC, C, ldc, sftA, sftB, *alpha, *beta);
    timing_stop(timetmp, timer[3]);

    return timer;
}

} // namespace gemmul8

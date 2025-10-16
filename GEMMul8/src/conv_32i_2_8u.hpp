#pragma once
#include "common.hpp"
#include "table.hpp"

namespace oz2_util {

namespace {
__device__ __forceinline__ uint8_t mod_reduce(int32_t x,    // input
                                              int32_t invp, // floor(2^32 / p - 1)
                                              uint8_t p)    // modulus
{
    int32_t quot = __mulhi(x, invp); // the most significant 32-bit of x*invm
    x -= quot * p;
    int32_t ge = (x - p) >> 31;
    int32_t lt = x >> 31;
    x -= (ge ^ -1) & p;
    x += lt & p;
    return static_cast<uint8_t>(x);
};
} // namespace

__global__ void conv_32i_2_8u_256_kernel(const size_t sizeC,                     // ((m * n + 15) >> 4) << 4; // multiple of 16
                                         const int32_t *const __restrict__ C32i, // input
                                         uint8_t *const __restrict__ C8u)        // output
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;

    int4 in = reinterpret_cast<const int4 *>(C32i)[idx];

    uchar4 out{static_cast<unsigned char>(in.x),
               static_cast<unsigned char>(in.y),
               static_cast<unsigned char>(in.z),
               static_cast<unsigned char>(in.w)};

    reinterpret_cast<uchar4 *>(C8u)[idx] = out;
}

__global__ void conv_32i_2_8u_not256_kernel(const size_t sizeC,                     // ((m * n + 15) >> 4) << 4; // multiple of 16
                                            const int32_t *const __restrict__ C32i, // input
                                            const uint8_t modulus,                  // <= 256
                                            const int32_t invm,                     // 2^32 / modulus - 1
                                            uint8_t *const __restrict__ C8u)        // output
{
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sizeC) return;
    
    int4 in = reinterpret_cast<const int4 *>(C32i)[idx];

    uchar4 out;
    out.x = mod_reduce(in.x, invm, modulus);
    out.y = mod_reduce(in.y, invm, modulus);
    out.z = mod_reduce(in.z, invm, modulus);
    out.w = mod_reduce(in.w, invm, modulus);

    reinterpret_cast<uchar4 *>(C8u)[idx] = out;
}

// interface!!
__inline__ void conv_32i_2_8u(const unsigned i,          //
                              const size_t sizeC,        // m*n/16*16
                              const int32_t *const C32i, // input
                              uint8_t *const C8u)        // output
{
    if (i == 0) {
        conv_32i_2_8u_256_kernel<<<oz2_const::grids_conv32i8u, oz2_const::threads_conv32i8u>>>(sizeC >> 2, C32i, C8u);
    } else {
        const uint8_t modulus = static_cast<uint8_t>(-oz2_table::moduli[i].z);
        const int32_t invm    = oz2_table::invm_32i[i - 1];
        conv_32i_2_8u_not256_kernel<<<oz2_const::grids_conv32i8u, oz2_const::threads_conv32i8u>>>(sizeC >> 2, C32i, modulus, invm, C8u);
    }
}

__global__ void conv_32i_2_8u_batched_kernel(const size_t sizeC,
                                             const int32_t *const __restrict__ C32i_all, // input
                                             uint8_t *const __restrict__ C8u_all)        // output
{
    const auto mod = blockIdx.y; // modulus index
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t sizeC4 = sizeC >> 2;
    if (idx >= sizeC4) return;

    const int32_t* C32i = C32i_all + static_cast<size_t>(mod) * sizeC;
    uint8_t*       C8u  = C8u_all  + static_cast<size_t>(mod) * sizeC;

    int4 in = reinterpret_cast<const int4 *>(C32i)[idx];

    uchar4 out;
    if (mod == 0) {
        out.x = static_cast<unsigned char>(in.x);
        out.y = static_cast<unsigned char>(in.y);
        out.z = static_cast<unsigned char>(in.z);
        out.w = static_cast<unsigned char>(in.w);
    } else {
        const uint8_t modulus = static_cast<uint8_t>(-oz2_table::moduli_dev[mod].z);
        const int32_t invm    = oz2_table::invm_32i_dev[mod - 1];
        out.x = mod_reduce(in.x, invm, modulus);
        out.y = mod_reduce(in.y, invm, modulus);
        out.z = mod_reduce(in.z, invm, modulus);
        out.w = mod_reduce(in.w, invm, modulus);
    }

    reinterpret_cast<uchar4*>(C8u)[idx] = out;
}

__inline__ void conv_32i_2_8u_batched(const unsigned num_moduli,
                                      const size_t sizeC,        // m*n/16*16
                                      const int32_t *const C32i, // input
                                      uint8_t *const C8u)        // output
{
    const int threads = oz2_const::threads_conv32i8u;
    const int blocks_x = static_cast<int>(((sizeC >> 2) + threads - 1) / threads);
    dim3 grid(blocks_x, num_moduli);
    conv_32i_2_8u_batched_kernel<<<grid, threads>>>(sizeC, C32i, C8u);
}

} // namespace oz2_util

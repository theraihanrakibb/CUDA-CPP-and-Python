// paged_attention.cu - paged attention on the GPU + a CPU reference check.
//
// The manager (kv_cache.h) stored this sequence's K/V in scattered physical
// blocks. This kernel reads them back through the block table and computes
// causal self-attention, exactly as a serving engine would - except we do
// one thread per query position for clarity (not for speed).
//
// Build:  make
// Run:    ./paged_attention

#include "kv_cache.h"
#include <cmath>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t e = (call);                                               \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(e));                                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// One CUDA block per query position. Thread 0 does the full attention for
// that position, reading keys/values via the per-sequence block table.
//   d_K/d_V : physical paged buffers
//   block_table : device array of this sequence's physical block ids
//   Q : query vector for THIS position, length head_dim
//   Out : written [head_dim]
__global__ void pagedAttention(const float* d_K, const float* d_V,
                               const int* block_table, int L, int block_size,
                               int head_dim, const float* Q, float* Out,
                               int qpos) {
    if (threadIdx.x != 0) return;
    extern __shared__ float scores[];
    float maxs = -1e30f;
    for (int j = 0; j < L; ++j) {
        int b = block_table[j / block_size];
        int slot = j % block_size;
        const float* k = d_K + ((size_t)b * block_size + slot) * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; ++d) s += Q[d] * k[d];
        scores[j] = s;
        if (s > maxs) maxs = s;
    }
    float sum = 0.0f;
    for (int j = 0; j < L; ++j) { scores[j] = expf(scores[j] - maxs); sum += scores[j]; }
    for (int d = 0; d < head_dim; ++d) Out[qpos * head_dim + d] = 0.0f;
    for (int j = 0; j < L; ++j) {
        int b = block_table[j / block_size];
        int slot = j % block_size;
        const float* v = d_V + ((size_t)b * block_size + slot) * head_dim;
        float w = scores[j] / sum;
        for (int d = 0; d < head_dim; ++d) Out[qpos * head_dim + d] += w * v[d];
    }
}

// CPU reference: same math, but K/V are dense [L][head_dim].
static void referenceAttention(const float* K, const float* V, int L, int head_dim,
                               float* Out) {
    for (int q = 0; q < L; ++q) {
        const float* Q = K + (size_t)q * head_dim;
        float maxs = -1e30f;
        std::vector<float> sc(L);
        for (int j = 0; j < L; ++j) {
            float s = 0.0f;
            for (int d = 0; d < head_dim; ++d)
                s += Q[d] * K[(size_t)j * head_dim + d];
            sc[j] = s;
            if (s > maxs) maxs = s;
        }
        float sum = 0.0f;
        for (int j = 0; j < L; ++j) { sc[j] = expf(sc[j] - maxs); sum += sc[j]; }
        for (int d = 0; d < head_dim; ++d) Out[(size_t)q * head_dim + d] = 0.0f;
        for (int j = 0; j < L; ++j) {
            float w = sc[j] / sum;
            for (int d = 0; d < head_dim; ++d)
                Out[(size_t)q * head_dim + d] += w * V[(size_t)j * head_dim + d];
        }
    }
}

int main() {
    const int block_size = 4;
    const int num_blocks = 32;
    const int num_heads  = 1;
    const int head_dim   = 8;
    const int L = 10;                 // sequence length (spans 3 blocks)

    PagedKVCache cache(block_size, num_blocks, num_heads, head_dim);
    int seq = cache.new_sequence();

    srand(42);
    std::vector<float> Khost((size_t)L * head_dim), Vhost((size_t)L * head_dim);
    for (int t = 0; t < L; ++t) {
        for (int d = 0; d < head_dim; ++d) {
            Khost[(size_t)t * head_dim + d] = (float)rand() / RAND_MAX - 0.5f;
            Vhost[(size_t)t * head_dim + d] = (float)rand() / RAND_MAX - 0.5f;
        }
        cache.append(seq, t, &Khost[(size_t)t * head_dim], &Vhost[(size_t)t * head_dim]);
    }

    // device copy of this sequence's block table
    std::vector<int> bt = cache.block_tables[seq];
    int* d_block_table;
    CUDA_CHECK(cudaMalloc(&d_block_table, bt.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_block_table, bt.data(), bt.size() * sizeof(int), cudaMemcpyHostToDevice));

    float* d_Out;
    CUDA_CHECK(cudaMalloc(&d_Out, (size_t)L * head_dim * sizeof(float)));

    // run paged attention, one kernel launch per query position (clear, not fast)
    for (int q = 0; q < L; ++q) {
        const float* Q = &Khost[(size_t)q * head_dim]; // self-query for the demo
        pagedAttention<<<1, 1, L * sizeof(float)>>>(
            cache.d_K, cache.d_V, d_block_table, L, block_size, head_dim, Q, d_Out, q);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpuOut((size_t)L * head_dim);
    CUDA_CHECK(cudaMemcpy(gpuOut.data(), d_Out, (size_t)L * head_dim * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> refOut((size_t)L * head_dim);
    referenceAttention(Khost.data(), Vhost.data(), L, head_dim, refOut.data());

    float maxErr = 0.0f;
    for (size_t i = 0; i < (size_t)L * head_dim; ++i)
        maxErr = fmaxf(maxErr, fabsf(gpuOut[i] - refOut[i]));
    printf("sequence length L=%d, blocks used=%d/%d\n", L, (int)bt.size(), num_blocks);
    printf("max error vs CPU reference: %e\n", maxErr);

    CUDA_CHECK(cudaFree(d_block_table));
    CUDA_CHECK(cudaFree(d_Out));
    return 0;
}

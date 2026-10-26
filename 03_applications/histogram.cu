// histogram.cu - 8-bit histogram using shared-memory privatization.
// Each block keeps a private histogram in __shared__ and only the per-bin
// totals are merged into global memory with atomics. This avoids one atomic
// per element (the slow part of a naive histogram).
//
// Build:  make
// Run:    ./histogram

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

const int BINS = 256;

__global__ void histogram(const unsigned char* data, unsigned int* hist, size_t n) {
    __shared__ unsigned int local[BINS];
    int i = threadIdx.x;
    // initialize shared histogram
    for (; i < BINS; i += blockDim.x)
        local[i] = 0;
    __syncthreads();

    // each thread strides through its portion of the data
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += blockDim.x * gridDim.x)
        atomicAdd(&local[data[idx]], 1u);

    __syncthreads();
    // merge into global histogram
    for (i = threadIdx.x; i < BINS; i += blockDim.x)
        atomicAdd(&hist[i], local[i]);
}

int main() {
    const size_t n = 1 << 22;   // ~4M samples
    size_t bytes = n * sizeof(unsigned char);
    unsigned char* h_data = (unsigned char*)malloc(bytes);
    unsigned int* h_hist = (unsigned int*)calloc(BINS, sizeof(unsigned int));
    unsigned int* h_ref  = (unsigned int*)calloc(BINS, sizeof(unsigned int));
    srand(99);
    for (size_t i = 0; i < n; ++i) h_data[i] = rand() % BINS;

    for (size_t i = 0; i < n; ++i) h_ref[h_data[i]]++;

    unsigned char* d_data;
    unsigned int* d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_hist, BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0, BINS * sizeof(unsigned int)));

    int threads = 256, blocks = 256;
    histogram<<<blocks, threads>>>(d_data, d_hist, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    unsigned int totalDiff = 0;
    for (int b = 0; b < BINS; ++b) totalDiff += abs((int)h_hist[b] - (int)h_ref[b]);
    printf("Total bin difference vs CPU: %u  (should be 0)\n", totalDiff);

    free(h_data); free(h_hist); free(h_ref);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_hist));
    return 0;
}

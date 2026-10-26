// monte_carlo.cu - estimate Pi by Monte Carlo sampling.
// Throw random points in the unit square; fraction inside the unit circle
// is Pi/4. Uses cuRAND-free XORSHIFT per-thread RNG for portability.
//
// Build:  make
// Run:    ./monte_carlo

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

// per-thread counter of points inside the quarter circle
__global__ void monteCarlo(unsigned long long* inCircle, unsigned long long total, unsigned long long seed) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long x = seed + tid * 6364136223846793005ULL + 1442695040888963407ULL;
    unsigned long long local = 0;

    // deterministic, independent LCG stream per thread
    unsigned long long state = x;
    for (unsigned long long i = 0; i < total; i += gridDim.x * blockDim.x) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float rx = (float)(state >> 32) / (float)(0xFFFFFFFFu);
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float ry = (float)(state >> 32) / (float)(0xFFFFFFFFu);
        if (rx * rx + ry * ry <= 1.0f) ++local;
    }
    // one atomic per thread (cheap: #threads << #samples)
    atomicAdd(inCircle, local);
}

int main() {
    const unsigned long long samples = 1ULL << 28;  // ~268M samples
    unsigned long long* d_inCircle;
    CUDA_CHECK(cudaMalloc(&d_inCircle, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_inCircle, 0, sizeof(unsigned long long)));

    int threads = 256;
    int blocks = 256;
    // samples handled per thread = samples / (blocks*threads)
    unsigned long long perThread = samples / (unsigned long long)(blocks * threads);
    monteCarlo<<<blocks, threads>>>(d_inCircle, perThread, 123456789ULL);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long h_in = 0;
    CUDA_CHECK(cudaMemcpy(&h_in, d_inCircle, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    unsigned long long total = perThread * (unsigned long long)(blocks * threads);
    double pi = 4.0 * (double)h_in / (double)total;
    printf("Pi estimate: %.6f  (samples=%llu, inside=%llu)\n", pi, total, h_in);

    CUDA_CHECK(cudaFree(d_inCircle));
    return 0;
}

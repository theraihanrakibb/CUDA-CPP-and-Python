// kernel_timing.cu - precise GPU timing with cudaEvent.
// Times a saxpy-like kernel across a sweep of problem sizes and reports
// achieved throughput. Uses events (not CPU wall-clock) for accuracy.
//
// Build:  make
// Run:    ./kernel_timing

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

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        y[i] = a * x[i] + y[i];
}

int main() {
    const int threads = 256;
    const int trials = 50;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("%-12s %-10s %-12s %-12s\n", "elements", "time(ms)", "GB/s", "GFLOP/s");

    for (int exp = 20; exp <= 28; exp += 2) {
        size_t n = (size_t)1 << exp;
        size_t bytes = n * sizeof(float);
        float *d_x, *d_y;
        CUDA_CHECK(cudaMalloc(&d_x, bytes));
        CUDA_CHECK(cudaMalloc(&d_y, bytes));
        CUDA_CHECK(cudaMemset(d_x, 1, bytes));
        CUDA_CHECK(cudaMemset(d_y, 2, bytes));

        int blocks = (n + threads - 1) / threads;

        // warmup
        saxpy<<<blocks, threads>>>(n, 2.0f, d_x, d_y);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int t = 0; t < trials; ++t)
            saxpy<<<blocks, threads>>>(n, 2.0f, d_x, d_y);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        double msPer = ms / trials;
        // read + write = 2*n floats moved; 2 flops per element (a*x + y)
        double gbps = (2.0 * bytes) / (msPer / 1000.0) / (1 << 30);
        double gflops = (2.0 * (double)n) / (msPer / 1000.0) / 1e9;
        printf("%-12zu %-10.3f %-12.1f %-12.1f\n", n, msPer, gbps, gflops);

        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}

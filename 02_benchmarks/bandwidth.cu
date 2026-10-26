// bandwidth.cu - measure device memory bandwidth.
// Reports GB/s for a device-to-device copy done two ways:
//   1) cudaMemcpy (DMA engine)
//   2) a naive copy kernel (each thread copies one float)
//
// Build:  make
// Run:    ./bandwidth

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__global__ void copyKernel(const float* in, float* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride)
        out[i] = in[i];
}

int main() {
    const size_t n = 1 << 26;            // 64M floats ~ 256 MB
    const size_t bytes = n * sizeof(float);
    const int trials = 20;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_in, 1, bytes));

    // 1) cudaMemcpy bandwidth
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = 0;
    for (int t = 0; t < trials; ++t) {
        CUDA_CHECK(cudaMemcpy(d_out, d_in, bytes, cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    // rough wall-clock for the whole loop
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int t = 0; t < trials; ++t)
        CUDA_CHECK(cudaMemcpy(d_out, d_in, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double gb = (double)bytes * trials / (1 << 30);
    printf("cudaMemcpy D2D : %7.2f GB/s  (%.2f ms for %d copies)\n",
           gb / (ms / 1000.0), ms, trials);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 2) kernel copy bandwidth
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int t = 0; t < trials; ++t)
        copyKernel<<<blocks, threads>>>(d_in, d_out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("kernel copy    : %7.2f GB/s  (%.2f ms for %d copies)\n",
           gb / (ms / 1000.0), ms, trials);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}

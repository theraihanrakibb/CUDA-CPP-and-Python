// occupancy.cu - inspect how block size affects GPU occupancy.
// Uses cudaOccupancyMaxActiveBlocksPerMultiprocessor to report, for a
// given kernel and block size, the maximum active blocks per SM and the
// resulting occupancy percentage.
//
// Build:  make
// Run:    ./occupancy

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

// A kernel that uses a moderate amount of shared memory so block size
// visibly trades off against per-block resources.
#define SHMEM (8 * 1024)
__global__ void workKernel(float* out) {
    __shared__ float buf[SHMEM / sizeof(float)];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.0f;
    for (int k = 0; k < 256; ++k)
        s += buf[threadIdx.x % (SHMEM / sizeof(float))] + (float)k;
    out[i] = s;
}

int main() {
    cudaDeviceProp prop;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, prop.multiProcessorCount * 1024 * sizeof(float)));

    printf("Device: %s  (%d SMs, %d threads/SM max)\n",
           prop.name, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);
    printf("%-10s %-10s %-14s %-12s\n", "blockSize", "activeBlks", "threadsActive", "occupancy%");

    for (int blockSize = 32; blockSize <= 1024; blockSize += 32) {
        int maxActiveBlocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks, workKernel, blockSize, SHMEM));
        int threadsActive = maxActiveBlocks * blockSize;
        float occupancy = (float)threadsActive / prop.maxThreadsPerMultiProcessor * 100.0f;
        printf("%-10d %-10d %-14d %-11.1f\n",
               blockSize, maxActiveBlocks, threadsActive, occupancy);
    }

    CUDA_CHECK(cudaFree(d_out));
    return 0;
}

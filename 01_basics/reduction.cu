// reduction.cu — parallel reduction in shared memory.
// Sums N floats on the GPU. Each block reduces to one partial sum,
// then the host sums the partials.
//
// Build:  make
// Run:    ./reduction

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

// Interleaved (stride) reduction with in-warp divergence avoided by
// fully unrolled loop. Each block produces one partial sum in out[blockIdx.x].
__global__ void reduce(const float* in, float* out, size_t n) {
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (; i < n; i += stride)
        sum += in[i];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        out[blockIdx.x] = sdata[0];
}

int main() {
    const size_t n = 1 << 24;          // 16M elements
    const int threads = 256;
    const int blocks = 256;
    size_t inBytes = n * sizeof(float);
    size_t outBytes = blocks * sizeof(float);

    float *h_in = (float*)malloc(inBytes);
    float *h_out = (float*)malloc(outBytes);
    for (size_t i = 0; i < n; ++i)
        h_in[i] = 1.0f;                 // sum should equal n

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, inBytes));
    CUDA_CHECK(cudaMalloc(&d_out, outBytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, inBytes, cudaMemcpyHostToDevice));

    reduce<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, outBytes, cudaMemcpyDeviceToHost));

    double gpu_sum = 0.0;
    for (int i = 0; i < blocks; ++i)
        gpu_sum += h_out[i];

    double cpu_sum = 0.0;
    for (size_t i = 0; i < n; ++i)
        cpu_sum += h_in[i];

    printf("GPU sum: %.1f  CPU sum: %.1f  error: %e\n",
           gpu_sum, cpu_sum, fabs(gpu_sum - cpu_sum));

    free(h_in); free(h_out);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}

// image_blur.cu - 3x3 box blur on a grayscale image.
// Each thread loads its pixel and averages its 3x3 neighborhood.
//
// Build:  make
// Run:    ./image_blur

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

__global__ void boxBlur(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int sum = 0, cnt = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                sum += in[ny * w + nx];
                ++cnt;
            }
        }
    }
    out[y * w + x] = (unsigned char)(sum / cnt);
}

int main() {
    const int w = 1024, h = 768;
    size_t bytes = (size_t)w * h;

    unsigned char *h_in = (unsigned char*)malloc(bytes);
    unsigned char *h_out = (unsigned char*)malloc(bytes);
    unsigned char *h_ref = (unsigned char*)malloc(bytes);
    srand(7);
    for (size_t i = 0; i < bytes; ++i) h_in[i] = rand() % 256;

    // CPU reference
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int sum = 0, cnt = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                        sum += h_in[ny * w + nx]; ++cnt;
                    }
                }
            h_ref[y * w + x] = (unsigned char)(sum / cnt);
        }

    unsigned char *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    boxBlur<<<grid, block>>>(d_in, d_out, w, h);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    int diff = 0;
    for (size_t i = 0; i < bytes; ++i) diff += abs((int)h_out[i] - (int)h_ref[i]);
    printf("Total abs diff vs CPU: %d  (avg %.4f per pixel)\n", diff, (float)diff / bytes);

    free(h_in); free(h_out); free(h_ref);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}

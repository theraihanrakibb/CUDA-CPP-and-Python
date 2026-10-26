# Project 1 - Learning Examples

Three small CUDA programs that build the core mental model:

| File            | What it teaches                                             |
|-----------------|-------------------------------------------------------------|
| vector_add.cu   | Kernels, grid/block indexing, memory copy, error checks     |
| matrix_mul.cu   | Tiled multiply with __shared__ memory + coalescing          |
| reduction.cu    | Parallel reduction in shared memory, partial sums           |

## Build & run

Requires the CUDA toolkit (nvcc). On Linux/WSL:

    make
    make run

Or compile a single file directly:

    nvcc -O3 -arch=sm_60 vector_add.cu -o vector_add
    ./vector_add

(-arch=sm_60 targets Pascal; raise it to match your GPU, e.g. sm_75, sm_86.)

## Expected output

- vector_add: Max error: 0.000000e+00
- matrix_mul: Max error vs CPU: ~1e-5 (float precision)
- reduction: GPU sum equal to CPU sum (= 16777216)

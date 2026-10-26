# Makefile for the learning-examples project.
# Requires the CUDA toolkit (nvcc) on PATH.
#   make          build all three programs
#   make run      build and run each
#   make clean    remove binaries

NVCC      ?= nvcc
CUDAFLAGS ?= -O3 -arch=sm_60
TARGETS   := vector_add matrix_mul reduction

.PHONY: all run clean
all: $(TARGETS)

%: %.cu
	$(NVCC) $(CUDAFLAGS) -o $@ $<

run: all
	./vector_add
	./matrix_mul
	./reduction

clean:
	rm -f $(TARGETS)

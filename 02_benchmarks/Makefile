# Makefile for the benchmarks project.
# Requires the CUDA toolkit (nvcc) on PATH.
#   make          build all three programs
#   make run      build and run each
#   make clean    remove binaries

NVCC      ?= nvcc
CUDAFLAGS ?= -O3 -arch=sm_60
TARGETS   := bandwidth occupancy kernel_timing

.PHONY: all run clean
all: $(TARGETS)

%: %.cu
	$(NVCC) $(CUDAFLAGS) -o $@ $<

run: all
	./bandwidth
	./occupancy
	./kernel_timing

clean:
	rm -f $(TARGETS)

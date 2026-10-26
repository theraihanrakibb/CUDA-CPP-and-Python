# Makefile for the applications project.
# Requires the CUDA toolkit (nvcc) on PATH.
#   make          build all three programs
#   make run      build and run each
#   make clean    remove binaries

NVCC      ?= nvcc
CUDAFLAGS ?= -O3 -arch=sm_60
TARGETS   := image_blur histogram monte_carlo

.PHONY: all run clean
all: $(TARGETS)

%: %.cu
	$(NVCC) $(CUDAFLAGS) -o $@ $<

run: all
	./image_blur
	./histogram
	./monte_carlo

clean:
	rm -f $(TARGETS)

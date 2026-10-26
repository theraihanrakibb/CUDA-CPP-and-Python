# Makefile for the C++ KV-cache project.
NVCC      ?= nvcc
CUDAFLAGS ?= -O3 -arch=sm_60
TARGET    := paged_attention

.PHONY: all run clean
all: $(TARGET)

$(TARGET): paged_attention.cu kv_cache.h
	$(NVCC) $(CUDAFLAGS) -o $@ paged_attention.cu

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)

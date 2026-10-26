# Makefile for the C++ train+infer project.
NVCC      ?= nvcc
CUDAFLAGS ?= -O3 -arch=sm_60
TARGET    := tiny_transformer

.PHONY: all run clean
all: $(TARGET)

$(TARGET): tiny_transformer.cu
	$(NVCC) $(CUDAFLAGS) -o $@ tiny_transformer.cu

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)

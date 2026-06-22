NVCC ?= nvcc
TARGET = benchmark_cuda.exe
NVCCFLAGS = -O2

.PHONY: all clean run

all: $(TARGET)

$(TARGET): benchmark_cuda.cu mlp_cuda.cu kernels.cuh
	$(NVCC) $(NVCCFLAGS) benchmark_cuda.cu mlp_cuda.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	-cmd /C "if exist $(TARGET) del /Q $(TARGET)"

CC = gcc
TARGET = benchmark_openmp.exe
CFLAGS = -O2 -std=c11 -Wall -Wextra -fopenmp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): benchmark_openmp.c openmp_preprocessing.c
	$(CC) $(CFLAGS) benchmark_openmp.c -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	-cmd /C "if exist $(TARGET) del /Q $(TARGET)"

#include "kernels.cuh"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CUDA_BLOCK_SIZE 256
#define CUDA_REPETITIONS 5
#define CUDA_BATCH_OPTIONS 4

static int check_cuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(status));
        return 0;
    }
    return 1;
}

static float* allocate_float_array(size_t size) {
    float* values = (float*)malloc(size * sizeof(float));
    if (values == NULL) {
        fprintf(stderr, "No fue posible reservar memoria host.\n");
    }
    return values;
}

static void fill_deterministic_vector(float* values, size_t size, float scale, int offset) {
    size_t i;
    for (i = 0u; i < size; ++i) {
        const int raw = (int)((i * 37u + (size_t)offset * 17u) % 1000u);
        values[i] = ((float)raw / 1000.0f - 0.5f) * scale;
    }
}

static double now_ms(void) {
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
}

static void matmul_cpu(const float* input, const float* weights, float* output, int batch_size, int in_dim, int out_dim) {
    int row;
    for (row = 0; row < batch_size; ++row) {
        int col;
        for (col = 0; col < out_dim; ++col) {
            float acc = 0.0f;
            int k;
            for (k = 0; k < in_dim; ++k) {
                acc += input[row * in_dim + k] * weights[k * out_dim + col];
            }
            output[row * out_dim + col] = acc;
        }
    }
}

static void bias_relu_cpu(float* values, const float* bias, int total_values, int units) {
    int i;
    for (i = 0; i < total_values; ++i) {
        const float value = values[i] + bias[i % units];
        values[i] = value > 0.0f ? value : 0.0f;
    }
}

static void bias_sigmoid_cpu(float* values, const float* bias, int total_values, int units) {
    int i;
    for (i = 0; i < total_values; ++i) {
        const float value = values[i] + bias[i % units];
        values[i] = 1.0f / (1.0f + expf(-value));
    }
}

static float binary_cross_entropy_cpu(const float* predictions, const float* labels, int total_values) {
    const float eps = 1.0e-7f;
    float total = 0.0f;
    int i;

    for (i = 0; i < total_values; ++i) {
        float pred = predictions[i];
        const float label = labels[i];
        if (pred < eps) {
            pred = eps;
        }
        if (pred > 1.0f - eps) {
            pred = 1.0f - eps;
        }
        total += -(label * logf(pred) + (1.0f - label) * logf(1.0f - pred));
    }

    return total / (float)total_values;
}

static void forward_cpu(
    const float* input,
    const float* w1,
    const float* b1,
    const float* w2,
    const float* b2,
    const float* w3,
    const float* b3,
    float* z1,
    float* z2,
    float* output,
    int batch_size
) {
    matmul_cpu(input, w1, z1, batch_size, CUDA_INPUT_DIM, CUDA_HIDDEN1);
    bias_relu_cpu(z1, b1, batch_size * CUDA_HIDDEN1, CUDA_HIDDEN1);
    matmul_cpu(z1, w2, z2, batch_size, CUDA_HIDDEN1, CUDA_HIDDEN2);
    bias_relu_cpu(z2, b2, batch_size * CUDA_HIDDEN2, CUDA_HIDDEN2);
    matmul_cpu(z2, w3, output, batch_size, CUDA_HIDDEN2, CUDA_OUTPUT_DIM);
    bias_sigmoid_cpu(output, b3, batch_size * CUDA_OUTPUT_DIM, CUDA_OUTPUT_DIM);
}

static double measure_cpu_ms(
    int batch_size,
    int repetitions,
    const float* input,
    const float* labels,
    const float* w1,
    const float* b1,
    const float* w2,
    const float* b2,
    const float* w3,
    const float* b3,
    float* loss_out
) {
    float* z1 = allocate_float_array((size_t)batch_size * CUDA_HIDDEN1);
    float* z2 = allocate_float_array((size_t)batch_size * CUDA_HIDDEN2);
    float* output = allocate_float_array((size_t)batch_size * CUDA_OUTPUT_DIM);
    double start;
    double end;
    float loss_acc = 0.0f;
    int rep;

    if (z1 == NULL || z2 == NULL || output == NULL) {
        free(z1);
        free(z2);
        free(output);
        return -1.0;
    }

    start = now_ms();
    for (rep = 0; rep < repetitions; ++rep) {
        forward_cpu(input, w1, b1, w2, b2, w3, b3, z1, z2, output, batch_size);
        loss_acc += binary_cross_entropy_cpu(output, labels, batch_size);
    }
    end = now_ms();

    *loss_out = loss_acc / (float)repetitions;

    free(z1);
    free(z2);
    free(output);
    return end - start;
}

static void free_cuda_buffers(float* d_input, float* d_labels, float* d_losses, CudaMlpWeights* weights, CudaMlpActivations* activations) {
    cudaFree(d_input);
    cudaFree(d_labels);
    cudaFree(d_losses);
    cudaFree(weights->w1);
    cudaFree(weights->b1);
    cudaFree(weights->w2);
    cudaFree(weights->b2);
    cudaFree(weights->w3);
    cudaFree(weights->b3);
    cudaFree(activations->z1);
    cudaFree(activations->z2);
    cudaFree(activations->output);
}

static double measure_cuda_ms(
    int batch_size,
    int repetitions,
    const float* input,
    const float* labels,
    const float* w1,
    const float* b1,
    const float* w2,
    const float* b2,
    const float* w3,
    const float* b3,
    float* loss_out
) {
    float* d_input = NULL;
    float* d_labels = NULL;
    float* d_losses = NULL;
    CudaMlpWeights weights;
    CudaMlpActivations activations;
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsed_ms = 0.0f;
    float* host_losses = NULL;
    size_t i;
    int rep;

    memset(&weights, 0, sizeof(weights));
    memset(&activations, 0, sizeof(activations));

    if (!check_cuda(cudaMalloc((void**)&d_input, (size_t)batch_size * CUDA_INPUT_DIM * sizeof(float)), "cudaMalloc input")
        || !check_cuda(cudaMalloc((void**)&d_labels, (size_t)batch_size * sizeof(float)), "cudaMalloc labels")
        || !check_cuda(cudaMalloc((void**)&d_losses, (size_t)batch_size * sizeof(float)), "cudaMalloc losses")
        || !check_cuda(cudaMalloc((void**)&weights.w1, (size_t)CUDA_INPUT_DIM * CUDA_HIDDEN1 * sizeof(float)), "cudaMalloc w1")
        || !check_cuda(cudaMalloc((void**)&weights.b1, (size_t)CUDA_HIDDEN1 * sizeof(float)), "cudaMalloc b1")
        || !check_cuda(cudaMalloc((void**)&weights.w2, (size_t)CUDA_HIDDEN1 * CUDA_HIDDEN2 * sizeof(float)), "cudaMalloc w2")
        || !check_cuda(cudaMalloc((void**)&weights.b2, (size_t)CUDA_HIDDEN2 * sizeof(float)), "cudaMalloc b2")
        || !check_cuda(cudaMalloc((void**)&weights.w3, (size_t)CUDA_HIDDEN2 * CUDA_OUTPUT_DIM * sizeof(float)), "cudaMalloc w3")
        || !check_cuda(cudaMalloc((void**)&weights.b3, (size_t)CUDA_OUTPUT_DIM * sizeof(float)), "cudaMalloc b3")
        || !check_cuda(cudaMalloc((void**)&activations.z1, (size_t)batch_size * CUDA_HIDDEN1 * sizeof(float)), "cudaMalloc z1")
        || !check_cuda(cudaMalloc((void**)&activations.z2, (size_t)batch_size * CUDA_HIDDEN2 * sizeof(float)), "cudaMalloc z2")
        || !check_cuda(cudaMalloc((void**)&activations.output, (size_t)batch_size * CUDA_OUTPUT_DIM * sizeof(float)), "cudaMalloc output")) {
        free_cuda_buffers(d_input, d_labels, d_losses, &weights, &activations);
        return -1.0;
    }

    if (!check_cuda(cudaMemcpy(d_input, input, (size_t)batch_size * CUDA_INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice), "copy input")
        || !check_cuda(cudaMemcpy(d_labels, labels, (size_t)batch_size * sizeof(float), cudaMemcpyHostToDevice), "copy labels")
        || !check_cuda(cudaMemcpy(weights.w1, w1, (size_t)CUDA_INPUT_DIM * CUDA_HIDDEN1 * sizeof(float), cudaMemcpyHostToDevice), "copy w1")
        || !check_cuda(cudaMemcpy(weights.b1, b1, (size_t)CUDA_HIDDEN1 * sizeof(float), cudaMemcpyHostToDevice), "copy b1")
        || !check_cuda(cudaMemcpy(weights.w2, w2, (size_t)CUDA_HIDDEN1 * CUDA_HIDDEN2 * sizeof(float), cudaMemcpyHostToDevice), "copy w2")
        || !check_cuda(cudaMemcpy(weights.b2, b2, (size_t)CUDA_HIDDEN2 * sizeof(float), cudaMemcpyHostToDevice), "copy b2")
        || !check_cuda(cudaMemcpy(weights.w3, w3, (size_t)CUDA_HIDDEN2 * CUDA_OUTPUT_DIM * sizeof(float), cudaMemcpyHostToDevice), "copy w3")
        || !check_cuda(cudaMemcpy(weights.b3, b3, (size_t)CUDA_OUTPUT_DIM * sizeof(float), cudaMemcpyHostToDevice), "copy b3")
        || !check_cuda(cudaEventCreate(&start), "create start event")
        || !check_cuda(cudaEventCreate(&stop), "create stop event")
        || !check_cuda(cudaEventRecord(start), "record start")) {
        free_cuda_buffers(d_input, d_labels, d_losses, &weights, &activations);
        return -1.0;
    }

    for (rep = 0; rep < repetitions; ++rep) {
        cuda_forward_pass(d_input, &weights, &activations, batch_size);
        binary_cross_entropy_kernel<<<(batch_size + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE>>>(
            activations.output,
            d_labels,
            d_losses,
            batch_size
        );
    }

    if (!check_cuda(cudaEventRecord(stop), "record stop")
        || !check_cuda(cudaEventSynchronize(stop), "synchronize stop")
        || !check_cuda(cudaGetLastError(), "CUDA kernel execution")
        || !check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed time")) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free_cuda_buffers(d_input, d_labels, d_losses, &weights, &activations);
        return -1.0;
    }

    host_losses = allocate_float_array((size_t)batch_size);
    if (host_losses == NULL) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free_cuda_buffers(d_input, d_labels, d_losses, &weights, &activations);
        return -1.0;
    }

    if (!check_cuda(cudaMemcpy(host_losses, d_losses, (size_t)batch_size * sizeof(float), cudaMemcpyDeviceToHost), "copy losses")) {
        free(host_losses);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free_cuda_buffers(d_input, d_labels, d_losses, &weights, &activations);
        return -1.0;
    }

    *loss_out = 0.0f;
    for (i = 0u; i < (size_t)batch_size; ++i) {
        *loss_out += host_losses[i] / (float)batch_size;
    }

    free(host_losses);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free_cuda_buffers(d_input, d_labels, d_losses, &weights, &activations);
    return (double)elapsed_ms;
}

static void write_row(FILE* csv, const char* version, int batch_size, int repetitions, double time_ms, double speedup) {
    fprintf(csv, "%s,%d,%d,%.3f,%.3f\n", version, batch_size, repetitions, time_ms, speedup);
}

int main(void) {
    const int batch_sizes[CUDA_BATCH_OPTIONS] = {1, 8, 16, 32};
    FILE* csv = fopen("../metrics/cuda_benchmark.csv", "w");
    float* w1 = allocate_float_array((size_t)CUDA_INPUT_DIM * CUDA_HIDDEN1);
    float* b1 = allocate_float_array((size_t)CUDA_HIDDEN1);
    float* w2 = allocate_float_array((size_t)CUDA_HIDDEN1 * CUDA_HIDDEN2);
    float* b2 = allocate_float_array((size_t)CUDA_HIDDEN2);
    float* w3 = allocate_float_array((size_t)CUDA_HIDDEN2 * CUDA_OUTPUT_DIM);
    float* b3 = allocate_float_array((size_t)CUDA_OUTPUT_DIM);
    int option;

    if (csv == NULL) {
        csv = fopen("metrics/cuda_benchmark.csv", "w");
    }

    if (csv == NULL || w1 == NULL || b1 == NULL || w2 == NULL || b2 == NULL || w3 == NULL || b3 == NULL) {
        fprintf(stderr, "No fue posible inicializar el benchmark CUDA.\n");
        free(w1);
        free(b1);
        free(w2);
        free(b2);
        free(w3);
        free(b3);
        if (csv != NULL) {
            fclose(csv);
        }
        return 1;
    }

    fill_deterministic_vector(w1, (size_t)CUDA_INPUT_DIM * CUDA_HIDDEN1, 0.04f, 1);
    fill_deterministic_vector(b1, (size_t)CUDA_HIDDEN1, 0.02f, 2);
    fill_deterministic_vector(w2, (size_t)CUDA_HIDDEN1 * CUDA_HIDDEN2, 0.04f, 3);
    fill_deterministic_vector(b2, (size_t)CUDA_HIDDEN2, 0.02f, 4);
    fill_deterministic_vector(w3, (size_t)CUDA_HIDDEN2 * CUDA_OUTPUT_DIM, 0.04f, 5);
    fill_deterministic_vector(b3, (size_t)CUDA_OUTPUT_DIM, 0.02f, 6);

    fprintf(csv, "version,batch_size,repetitions,time_ms,speedup\n");

    for (option = 0; option < CUDA_BATCH_OPTIONS; ++option) {
        const int batch_size = batch_sizes[option];
        float* input = allocate_float_array((size_t)batch_size * CUDA_INPUT_DIM);
        float* labels = allocate_float_array((size_t)batch_size);
        float cpu_loss = 0.0f;
        float cuda_loss = 0.0f;
        double cpu_ms;
        double cuda_ms;
        double speedup;
        int i;

        if (input == NULL || labels == NULL) {
            free(input);
            free(labels);
            fclose(csv);
            free(w1);
            free(b1);
            free(w2);
            free(b2);
            free(w3);
            free(b3);
            return 1;
        }

        fill_deterministic_vector(input, (size_t)batch_size * CUDA_INPUT_DIM, 1.0f, batch_size);
        for (i = 0; i < batch_size; ++i) {
            labels[i] = (float)(i % 2);
        }

        cpu_ms = measure_cpu_ms(batch_size, CUDA_REPETITIONS, input, labels, w1, b1, w2, b2, w3, b3, &cpu_loss);
        write_row(csv, "cpu", batch_size, CUDA_REPETITIONS, cpu_ms, 1.0);

        cuda_ms = measure_cuda_ms(batch_size, CUDA_REPETITIONS, input, labels, w1, b1, w2, b2, w3, b3, &cuda_loss);
        speedup = cuda_ms > 0.0 ? cpu_ms / cuda_ms : 0.0;
        write_row(csv, "cuda", batch_size, CUDA_REPETITIONS, cuda_ms, speedup);

        printf("batch=%d cpu_ms=%.3f cuda_ms=%.3f speedup=%.3f cpu_loss_avg=%.6f cuda_loss_avg=%.6f\n",
            batch_size,
            cpu_ms,
            cuda_ms,
            speedup,
            cpu_loss,
            cuda_loss);

        free(input);
        free(labels);
    }

    fclose(csv);
    free(w1);
    free(b1);
    free(w2);
    free(b2);
    free(w3);
    free(b3);
    printf("CSV generado: ../metrics/cuda_benchmark.csv\n");
    return 0;
}

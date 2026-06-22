#pragma once

#include <cuda_runtime.h>

#define CUDA_INPUT_DIM 6400
#define CUDA_HIDDEN1 256
#define CUDA_HIDDEN2 64
#define CUDA_OUTPUT_DIM 1

__global__ void matmul_kernel(
    const float* input,
    const float* weights,
    float* output,
    int batch_size,
    int in_dim,
    int out_dim
);

__global__ void bias_relu_kernel(
    float* values,
    const float* bias,
    int total_values,
    int units
);

__global__ void bias_kernel(
    float* values,
    const float* bias,
    int total_values,
    int units
);

__global__ void sigmoid_kernel(float* values, int total_values);

__global__ void binary_cross_entropy_kernel(
    const float* predictions,
    const float* labels,
    float* losses,
    int total_values
);

__global__ void output_gradient_kernel(
    const float* predictions,
    const float* labels,
    float* gradients,
    int total_values
);

__global__ void sgd_update_kernel(
    float* parameters,
    const float* gradients,
    float learning_rate,
    int total_values
);

typedef struct {
    float* w1;
    float* b1;
    float* w2;
    float* b2;
    float* w3;
    float* b3;
} CudaMlpWeights;

typedef struct {
    float* z1;
    float* z2;
    float* output;
} CudaMlpActivations;

void cuda_forward_pass(
    const float* d_input,
    const CudaMlpWeights* weights,
    CudaMlpActivations* activations,
    int batch_size
);

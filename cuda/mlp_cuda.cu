#include "kernels.cuh"

#include <math.h>

#define CUDA_BLOCK_SIZE 256

__global__ void matmul_kernel(
    const float* input,
    const float* weights,
    float* output,
    int batch_size,
    int in_dim,
    int out_dim
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * out_dim;
    if (index >= total) {
        return;
    }

    const int row = index / out_dim;
    const int col = index % out_dim;
    float acc = 0.0f;

    for (int k = 0; k < in_dim; ++k) {
        acc += input[row * in_dim + k] * weights[k * out_dim + col];
    }

    output[index] = acc;
}

__global__ void bias_relu_kernel(float* values, const float* bias, int total_values, int units) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_values) {
        return;
    }

    const int unit = index % units;
    const float value = values[index] + bias[unit];
    values[index] = value > 0.0f ? value : 0.0f;
}

__global__ void bias_kernel(float* values, const float* bias, int total_values, int units) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_values) {
        return;
    }

    values[index] += bias[index % units];
}

__global__ void sigmoid_kernel(float* values, int total_values) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_values) {
        return;
    }

    values[index] = 1.0f / (1.0f + expf(-values[index]));
}

__global__ void binary_cross_entropy_kernel(
    const float* predictions,
    const float* labels,
    float* losses,
    int total_values
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_values) {
        return;
    }

    const float eps = 1.0e-7f;
    const float pred = fminf(fmaxf(predictions[index], eps), 1.0f - eps);
    const float label = labels[index];
    losses[index] = -(label * logf(pred) + (1.0f - label) * logf(1.0f - pred));
}

__global__ void output_gradient_kernel(
    const float* predictions,
    const float* labels,
    float* gradients,
    int total_values
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_values) {
        return;
    }

    gradients[index] = predictions[index] - labels[index];
}

__global__ void sgd_update_kernel(
    float* parameters,
    const float* gradients,
    float learning_rate,
    int total_values
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_values) {
        return;
    }

    parameters[index] -= learning_rate * gradients[index];
}

void cuda_forward_pass(
    const float* d_input,
    const CudaMlpWeights* weights,
    CudaMlpActivations* activations,
    int batch_size
) {
    const int h1_total = batch_size * CUDA_HIDDEN1;
    const int h2_total = batch_size * CUDA_HIDDEN2;
    const int out_total = batch_size * CUDA_OUTPUT_DIM;

    matmul_kernel<<<(h1_total + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE>>>(
        d_input,
        weights->w1,
        activations->z1,
        batch_size,
        CUDA_INPUT_DIM,
        CUDA_HIDDEN1
    );
    bias_relu_kernel<<<(h1_total + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE>>>(
        activations->z1,
        weights->b1,
        h1_total,
        CUDA_HIDDEN1
    );

    matmul_kernel<<<(h2_total + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE>>>(
        activations->z1,
        weights->w2,
        activations->z2,
        batch_size,
        CUDA_HIDDEN1,
        CUDA_HIDDEN2
    );
    bias_relu_kernel<<<(h2_total + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE>>>(
        activations->z2,
        weights->b2,
        h2_total,
        CUDA_HIDDEN2
    );

    matmul_kernel<<<(out_total + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE>>>(
        activations->z2,
        weights->w3,
        activations->output,
        batch_size,
        CUDA_HIDDEN2,
        CUDA_OUTPUT_DIM
    );
    bias_kernel<<<(out_total + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE>>>(
        activations->output,
        weights->b3,
        out_total,
        CUDA_OUTPUT_DIM
    );
    sigmoid_kernel<<<(out_total + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE>>>(activations->output, out_total);
}

#include <cuda_runtime.h>
#include <stdio.h>

#define WIDTH 1024
#define HEIGHT 1024

// CUDA kernel for matrix transposition
__global__ void transposeMatrix(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        output[y * width + x] = input[x * height + y];
    }
}

int main() {
    int width = WIDTH;
    int height = HEIGHT;
    size_t size = width * height * sizeof(float);

    // Allocate host memory
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    // Initialize the input matrix
    for (int i = 0; i < width * height; i++) {
        h_input[i] = (float)i;
    }

    // Allocate device memory
    float* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Define block and grid size
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    transposeMatrix<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Verify result
    int success = 1;
    for (int i = 0; i < width && success; i++) {
        for (int j = 0; j < height; j++) {
            if (h_output[i * height + j] != h_input[j * width + i]) {
                success = 0;
                break;
            }
        }
    }

    // Output result using printf
    if (success) {
        printf("Matrix transposition succeeded!\n");
    } else {
        printf("Matrix transposition failed!\n");
    }

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}

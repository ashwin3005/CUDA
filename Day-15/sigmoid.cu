#include <stdio.h>
#include <math.h>

// Sigmoid function kernel
__global__ void sigmoid_kernel(float *input_d, float *output_d, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output_d[idx] = 1.0f / (1.0f + expf(-input_d[idx]));
    }
}

int main() {
    int N = 10;  // Example size
    int bytes = N * sizeof(float);

    float *input_h, *output_h;

    input_h = (float*)malloc(bytes);
    output_h = (float*)malloc(bytes);
    
    float *input_d, *output_d;  // Device arrays

    // Initialize input array with some values
    for (int i = 0; i < N; i++) {
        input_h[i] = float(i);  // Example input values: 0, 1, 2, ...
    }

    // Allocate memory on device
    cudaMalloc((void**)&input_d, bytes);
    cudaMalloc((void**)&output_d, bytes);

    // Copy data from host to device
    cudaMemcpy(input_d, input_h, bytes, cudaMemcpyHostToDevice);

    // Define block size and grid size
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch the kernel
    sigmoid_kernel<<<gridSize, blockSize>>>(input_d, output_d, N);

    // Check for errors in kernel launch
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(output_h, output_d, bytes, cudaMemcpyDeviceToHost);

    // Print the results
    printf("Input values: ");
    for (int i = 0; i < N; i++) {
        printf("%f ", input_h[i]);
    }
    printf("\n");

    printf("Sigmoid outputs: ");
    for (int i = 0; i < N; i++) {
        printf("%f ", output_h[i]);
    }
    printf("\n");

    cudaFree(input_d);
    cudaFree(output_d);

    return 0;
}

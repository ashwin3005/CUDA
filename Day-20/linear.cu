#include <stdio.h>
#include <cuda_runtime.h>

__global__ void linear_layer_d(float* input_d, float* output_d, float* weights_d, float* bias_d, int input_size, int output_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < output_size) {
        output_d[i] = 0.0f;
        for (int j = 0; j < input_size; j++) {
            output_d[i] += input_d[j] * weights_d[i * input_size + j];
        }
        output_d[i] += bias_d[i];
    }
}

int main() {
    int input_size = 3;   
    int output_size = 2;  

    float input_h[3] = {1.0, 2.0, 3.0};

    float weights_h[6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};  

    float bias_h[2] = {0.1, 0.2};

    float output_h[2];

    float *input_d, *output_d, *weights_d, *bias_d;

    cudaMalloc((void**)&input_d, input_size * sizeof(float));
    cudaMalloc((void**)&output_d, output_size * sizeof(float));
    cudaMalloc((void**)&weights_d, input_size * output_size * sizeof(float));
    cudaMalloc((void**)&bias_d, output_size * sizeof(float));

    cudaMemcpy(input_d, input_h, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weights_d, weights_h, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_d, bias_h, output_size * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 2;
    int grid_size = (output_size + block_size - 1) / block_size;

    // Launch the kernel
    linear_layer_d<<<grid_size, block_size>>>(input_d, output_d, weights_d, bias_d, input_size, output_size);

    // Copy the result from device to host
    cudaMemcpy(output_h, output_d, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output: ");
    for (int i = 0; i < output_size; i++) {
        printf("%f ", output_h[i]);
    }
    printf("\n");

    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(weights_d);
    cudaFree(bias_d);

    return 0;
}

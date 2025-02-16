#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Leaky ReLU kernel
__global__ void leakyReLU_kernel(float *data_d, float alpha, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data_d[idx] = (data_d[idx] > 0) ? data_d[idx] : alpha * data_d[idx];
    }
}

int main() {
    int n = 16; 
    float alpha = 0.01f; // Leaky ReLU slope

    float *data_h = (float *)malloc(n * sizeof(float)); 
    float *data_d; 
    srand(time(NULL)); 

    for (int i = 0; i < n; i++) {
        data_h[i] = (float)rand()/float(RAND_MAX) * 100.0f; 
    }

    cudaMalloc((void **)&data_d, n * sizeof(float));
    cudaMemcpy(data_d, data_h, n * sizeof(float), cudaMemcpyHostToDevice);


    leakyReLU_kernel<<<(n + 255) / 256, 256>>>(data_d, alpha, n);
    

    cudaMemcpy(data_h, data_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i<n; i++){
        printf("%f ", data_d[i]);
    }

    // free 
    cudaFree(data_d);
    free(data_h);

    return 0;
}

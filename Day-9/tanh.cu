#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 12

// CUDA kernel for tanh activation function
__global__ void tanhActivation(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float exp2x = expf(2 * x);
        output[idx] = (exp2x - 1.0f) / (exp2x + 1.0f);
    }
}

void printArray(float *arr, int size){
    for(int i=0; i<size; i++){
        printf("%f ", arr[i]);
    }printf("\n");
}
int main(){
    size_t bytes = N * sizeof(float);    

    float *vec_h, *res_h;
    
    vec_h = (float*)malloc(bytes);
    res_h = (float*)malloc(bytes);

    srand(time(NULL));
    for(int i=0; i<N; i++){
        vec_h[i] = ((float)rand() / (float)(RAND_MAX)) * 20.0f - 10.0f;  
    }

    float *vec_d, *res_d;
    cudaMalloc((void**)&vec_d,bytes);
    cudaMalloc((void**)&res_d, bytes);

    cudaMemcpy(vec_d, vec_h, bytes, cudaMemcpyHostToDevice);

    dim3 THREADS(32, 1, 1);
    dim3 BLOCKS(N + THREADS.x / float(THREADS.x), 1, 1);

    // launching the kernel
    tanhActivation<<<BLOCKS, THREADS>>>(vec_d, res_d, N);

    cudaMemcpy(res_h, res_d, bytes, cudaMemcpyDeviceToHost);

    printf("vector:\n");
    printArray(vec_h, N);
    printf("vector after tanh:\n");
    printArray(res_h, N);
    return 0;
}

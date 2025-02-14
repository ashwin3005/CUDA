#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


// simple CUDA kernel for vector addition
__global__ void vectorAdd(float *A, float *B, float *C, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        C[idx] = A[idx] + B[idx];
    }
}

int main(){
    // number of elements
    int n = 10;

    float *A_h, *B_h, *result_h;      // pointers for the vectors in the host  (we will allocate the memory in the CPU)
    float *A_d, *B_d, *result_d;      // pointers for the vectors in the device (we will allocate the memory in the GPU)

    // Dynamic memory allocation for the vectors on the host
    A_h = (float*)malloc(n * sizeof(float));   // necessary to cast
    B_h = (float*)malloc(n * sizeof(float));
    result_h = (float*)malloc(n * sizeof(float));

    // initialize the vectors on the host with values
    for(int i = 0; i<n; i++){
        A_h[i] = i;
        B_h[i] = i;
    }

    // Dynamic memory allocation on GPU (device)
    cudaMalloc((void**)&A_d, n * sizeof(float));
    cudaMalloc((void**)&B_d, n * sizeof(float));
    cudaMalloc((void**)&result_d, n * sizeof(float));

    // copy the data to device
    cudaMemcpy(A_d, A_h, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, n * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 1024;  // number of threads :)   (max limit is 1024)
    int grid_size = (n/block_size) + 1;   // calcualate number of blocks

    // Launch the kernel
    vectorAdd<<<grid_size, block_size>>>(A_d, B_d, result_d, n);   

    // copy the result vector to the host from the device
    cudaMemcpy(result_h, result_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    // printing the result :)
    printf("idx A + B = result\n");
    for(int i=0; i<n; i++){
        printf("%d | %f + %f = %f\n", i, A_h[i], B_h[i], result_h[i]);
    }

    free(A_h);
    free(B_h);
    free(result_h);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(result_d);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>

__global__ void vecAdd(float *A, float *B, float *C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row * gridDim.x * blockDim.x + col;   // calculating i
    if(i<N){
       C[i] = A[i] + B[i];
    }  
}

int main(){
    
    const int N = 13;

    float *A_h, *B_h, *C_h; // arrays that will be allocated on the host.
    
    // allocating memoray in the host.
    A_h = (float*)malloc(N * sizeof(float));
    B_h = (float*)malloc(N * sizeof(float));
    C_h = (float*)malloc(N * sizeof(float));

    // initialising the values for the arrays;
    for(int i=0; i<N; i++){
        A_h[i] = i;
        B_h[i] = i + 3;
    }

    float *A_d, *B_d, *C_d; // pointers for arrays that will be allocated on the device.
    cudaMalloc(&A_d, N * sizeof(float));
    cudaMalloc(&B_d, N * sizeof(float));
    cudaMalloc(&C_d, N * sizeof(float));

    // copy data to device
    cudaMemcpy(A_d, A_h, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // defining the gridDim (no. of blocks) and blockDim (no. of threads per block)
    dim3 threadsPerBlock(2, 2);

    dim3 blocksPerGrid((N + threadsPerBlock.x * threadsPerBlock.y - 1) / (threadsPerBlock.x * threadsPerBlock.y));

    //launch the kernel
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);
    
    // copy back the data from the device to the host 
    cudaMemcpy(C_h, C_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    // print the result
    for(int i = 0; i<N; i++){
        printf("%f ", C_h[i]);
    }printf("\n");

    free(A_h);
    free(B_h);
    free(C_h);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    return 0;
}

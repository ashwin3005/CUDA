#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"

__global__ void squared_matmul(float *A, float *B, float *C, int width){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row<width && col < width){
        float dotprod = 0.0f;
        for(int i=0; i<width; i++){
            dotprod += A[row*width + i] * B[i*width + col];
        }C[row*width + col] =  dotprod;
    }
}

__global__ void unco_squared_matmul(float *A, float *B, float *C, int width){
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    if (row<width && col < width){
        float dotprod = 0.0f;
        for(int i=0; i<width; i++){
            dotprod += A[row*width + i] * B[i*width + col];
        }C[row*width + col] =  dotprod;
    }
}

struct timer t;

int main(){

    int N = 10000; // 10k
    int bytes = N * N * sizeof(float);
    srand(time(NULL));

    float *A, *B, *C;

    A = (float*)malloc(bytes);
    B = (float*)malloc(bytes);
    C = (float*)malloc(bytes);

    for(int i=0; i<N*N; i++){
        A[i] = ((float)rand()/ (float)RAND_MAX) * 200.0f - 100.0f;
        B[i] = ((float)rand()/ (float)RAND_MAX) * 200.0f - 100.0f;
    }

    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, bytes);
    cudaMalloc((void**)&B_d, bytes);
    cudaMalloc((void**)&C_d, bytes);

    cudaMemcpy(A_d, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, bytes, cudaMemcpyHostToDevice);

    // launch kernel
    dim3 THREADS(32, 32, 1);
    dim3 BLOCKS((N + THREADS.x - 1/THREADS.x),(N + THREADS.x - 1/THREADS.x),1);

    start_timer(&t);
    squared_matmul<<<BLOCKS, THREADS>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();
    stop_timer(&t);

    printf("Time taken for mat-mul in row major %f seconds\n", time_diff(&t));

    start_timer(&t);
    unco_squared_matmul<<<BLOCKS, THREADS>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();
    stop_timer(&t);

    printf("Time taken for mat-mul in col major %f seconds\n", time_diff(&t));

    free(A);
    free(B);
    free(C);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    return 0;
}
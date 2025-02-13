#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

__global__ void matrixMultiplication(float *A, float *B, float *result, int A_rows, int A_cols, int B_rows, int B_cols){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int result_height = A_rows;
    int result_width = B_cols;

        // result matrix = A_row X B_col = 4 X 6

    if(row < result_height && col < result_width){   // boundry check
        // dot product
        float sum = 0.0f;
        for(int i=0; i<A_cols; i++){
            sum += A[row * A_cols + i] * B[i * B_cols + col];
        }result[row*B_cols + col] = sum;
    }
}

int main(){
    int A_rows = 4;
    int A_cols = 3;
    int B_rows = 3;
    int B_cols = 6;

    size_t A_bytes = sizeof(float) * A_rows * A_cols;
    size_t B_bytes = sizeof(float) * B_rows * B_cols;
    size_t result_bytes = sizeof(float) * A_rows * B_cols;

    float *A, *B, *result;

    A = (float*)malloc(A_bytes);
    B = (float*)malloc(B_bytes);
    result = (float*)malloc(result_bytes);

    float *A_d, *B_d, *result_d;
    cudaMalloc((void**)&A_d, A_bytes);
    cudaMalloc((void**)&B_d, B_bytes);
    cudaMalloc((void**)&result_d, result_bytes);

    for(int i=0; i<A_rows*A_cols; i++){
        A[i] = i+1;
    }
    for(int i=0; i<B_rows*B_cols; i++){
        B[i] = 2*i+1;
    }

    printf("A Matrix: \n");
    printMatrix(A, A_rows, A_cols);
    printf("B Matrix: \n");
    printMatrix(B, B_rows, B_cols);

    cudaMemcpy(A_d, A, A_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, B_bytes, cudaMemcpyHostToDevice);

    // kernel launch
    dim3 THREADS(32, 32, 1);
    dim3 BLOCKS((B_cols + THREADS.x - 1)/THREADS.x, (A_rows + THREADS.y - 1)/THREADS.y, 1);

    matrixMultiplication<<<BLOCKS, THREADS>>>(A_d, B_d, result_d, A_rows, A_cols, B_rows, B_cols);

    cudaMemcpy(result, result_d, result_bytes, cudaMemcpyDeviceToHost);

    printf("Result: \n");
    printMatrix(result, A_rows, B_cols);

    free(A);
    free(B);
    free(result);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(result_d);
    return 0;
}


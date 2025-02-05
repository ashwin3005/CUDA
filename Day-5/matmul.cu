#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void matMul(int *A, int *B, int *C, int width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if(row < width && col < width){
        int sum = 0;
        // Dot product of the row and column
        for(int i = 0; i < width; i++){
            sum += A[row * width + i] * B[i * width + col];
        }
        C[row * width + col] = sum;
    }
}

void printMatrix(int *A, int rows, int cols);

int main(){
    int rows = 3; // height
    int cols = 3; // width
    int width = rows; // we are using square matrix :)

    srand(time(NULL));  // seeding the rand() with time

    // memory allocation for the host
    int *A = (int*)malloc(rows * cols * sizeof(int));
    int *B = (int*)malloc(rows * cols * sizeof(int));
    int *C = (int*)malloc(rows * cols * sizeof(int));  // Result matrix

    // Assign values to A and B
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            A[i * cols + j] = (rand()%10) + 1;    // initialise with a random number between 1 to 10
            B[i * cols + j] = (rand()%10) + 1;
        }
    }

    printf("Matrix A:\n");
    printMatrix(A, rows, cols);

    printf("Matrix B:\n");
    printMatrix(B, rows, cols);

    // Allocate device memory for matrices A, B, and C
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, rows * cols * sizeof(int));
    cudaMalloc((void**)&d_B, rows * cols * sizeof(int));
    cudaMalloc((void**)&d_C, rows * cols * sizeof(int));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A, rows * cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, rows * cols * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16, 1);  // 16x16 block size  
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y, 1); // grid size

    // kernel launch
    matMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, width);

    // Copy the result back to host
    cudaMemcpy(C, d_C, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result matrix C
    printf("Matrix C (Result):\n");
    printMatrix(C, rows, cols);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}

void printMatrix(int *A, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", A[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

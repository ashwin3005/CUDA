#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void matMul(int *A, int *B, int *C, int A_width, int B_width, int C_width){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < C_width && col < C_width){
        int sum = 0;
        for(int i=0; i<A_width; i++){  // A_width is the common dimension
            sum += A[row * A_width + i] * B[i * B_width + col];
        }
        C[row * C_width + col] = sum;
    }
}

void printMat(int *M, int rows, int cols);

int main(){
    int A_rows = 2;  
    int A_cols = 5;  
    int B_rows = 5;  
    int B_cols = 2;  
    int C_rows = A_rows;  
    int C_cols = B_cols;  
    
    // Ensure A_cols == B_rows for matrix multiplication to be valid
    if (A_cols != B_rows) {
        printf("Matrix dimensions do not match for multiplication.\n");
        return -1;
    }

    size_t A_bytes = A_rows * A_cols * sizeof(int);
    size_t B_bytes = B_rows * B_cols * sizeof(int);
    size_t C_bytes = C_rows * C_cols * sizeof(int);

    int *A, *B, *C;
    
    A = (int*)malloc(A_bytes);
    B = (int*)malloc(B_bytes);
    C = (int*)malloc(C_bytes);

    // Initialize matrices with some values
    for(int i=0; i<A_rows*A_cols; i++){
        A[i] = rand()%10;
    }
    for(int i=0; i<B_rows*B_cols; i++){
        B[i] = rand()%10;
    }

    int *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, A_bytes);
    cudaMalloc((void**)&B_d, B_bytes);
    cudaMalloc((void**)&C_d, C_bytes);

    cudaMemcpy(A_d, A, A_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, B_bytes, cudaMemcpyHostToDevice);

    printMat(A, A_rows, A_cols);
    printMat(B, B_rows, B_cols);

    
    dim3 THREADS(32, 32, 1);
    dim3 BLOCKS((C_cols + THREADS.x - 1) / THREADS.x, (C_rows + THREADS.y - 1) / THREADS.y);
    
    
    matMul<<<BLOCKS, THREADS>>>(A_d, B_d, C_d, A_cols, B_cols, C_cols);

    cudaMemcpy(C, C_d, C_bytes, cudaMemcpyDeviceToHost);

    printMat(C, C_rows, C_cols);
    
    free(A);
    free(B);
    free(C);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    return 0;
}

void printMat(int *M, int rows, int cols){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            printf("%d ", M[i*cols + j]);
        } 
        printf("\n");
    }
    printf("\n");
}

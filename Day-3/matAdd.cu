#include <stdio.h>
#include <cuda_runtime.h>

#define N 3  

__global__ void matrixAdd(int *A, int *B, int *C, int width) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (idx < width && idy < width) {
        int index = idy * width + idx;
        C[index] = A[index] + B[index];
    }
}

int main() {
    int width = N;
    int size = width * width * sizeof(int);


    int *A_h = (int*)malloc(size);
    int *B_h = (int*)malloc(size);
    int *C_h = (int*)malloc(size);

    
    for (int i = 0; i < width * width; i++) {
        A_h[i] = i + 1;
        B_h[i] = (i + 1) * 2;
    }

    int *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);


    dim3 block(16, 16);

    int gridX = (int)ceil((float)width / block.x);
    int gridY = (int)ceil((float)width / block.y);
    dim3 grid(gridX, gridY);

    
    matrixAdd<<<grid, block>>>(A_d, B_d, C_d, width);

    
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    
    printf("Result Matrix:\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", C_h[i * width + j]);
        }
        printf("\n");
    }

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}

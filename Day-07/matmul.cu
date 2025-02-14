#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

struct timer {
    struct timespec start_time, end_time;
};

void start_timer(struct timer *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->start_time);
}

void stop_timer(struct timer *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->end_time);
}

double time_diff(struct timer *t) {
    double diff = (t->end_time.tv_sec - t->start_time.tv_sec) +
        (t->end_time.tv_nsec - t->start_time.tv_nsec) / 1000000000.0;
    return diff;
}

__global__ void matMul(int *A, int *B, int *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        int sum = 0;
        for (int i = 0; i < width; i++) {
            sum += A[row * width + i] * B[i * width + col];
        }
        C[row * width + col] = sum;
    }
}

void matMul_CPU(int *A, int *B, int *C, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            C[i * width + j] = 0;
            for (int k = 0; k < width; k++) {
                C[i * width + j] += A[i * width + k] * B[k * width + j];
            }
        }
    }
}



int main() {
    int rows = 1024;
    int cols = 1024;
    int width = rows;  // Square matrix

    srand(time(NULL));   // seeding the rand() with time.

    // dynamic memeroy allocation
    int *A = (int*)malloc(rows * cols * sizeof(int));
    int *B = (int*)malloc(rows * cols * sizeof(int));
    int *C = (int*)malloc(rows * cols * sizeof(int));

    // Initialize matrices A and B with random values
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            A[i * cols + j] = (rand() % 10) + 1;
            B[i * cols + j] = (rand() % 10) + 1;
        }
    }

    // memory allocation on device (GPU)
    int *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, rows * cols * sizeof(int));
    cudaMalloc((void**)&B_d, rows * cols * sizeof(int));
    cudaMalloc((void**)&C_d, rows * cols * sizeof(int));

    // from host to GPU
    cudaMemcpy(A_d, A, rows * cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, rows * cols * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16, 1);  // 16 X 16 threads
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y, 1); // grid size

    struct timer t;

    // Matrix multiplication on GPU
    start_timer(&t);
    matMul<<<gridSize, blockSize>>>(A_d, B_d, C_d, width);
    cudaDeviceSynchronize();    // ensures that all the operations in the GPU is completed before we stop the timer.
    stop_timer(&t);
    printf("Matrix multiplication on GPU: %f seconds\n", time_diff(&t));

    // copy the result back to CPU
    cudaMemcpy(C, C_d, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

    // CPU Matrix multiplication
    int *C_CPU = (int*)malloc(rows * cols * sizeof(int));     // to results for the matMul_CPU
    start_timer(&t);
    matMul_CPU(A, B, C_CPU, width);
    stop_timer(&t);
    printf("Matrix multiplication on CPU: %f seconds\n", time_diff(&t));

    // verify the output
    char is_correct = 1; 
    for(int i=0; i<rows*cols; i++){
        if(C[i]!=C_CPU[i]){
            is_correct = 0;
            break;
        }
    }
    if (is_correct){
        printf("Results from the CPU and the GPU matches ✅\n");
    }else{
        printf("Results from the GPU and the GPU does not matches ❌ \n");
    }
    // Free memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A);
    free(B);
    free(C);
    free(C_CPU);

    return 0;
}

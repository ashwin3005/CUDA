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

// regular matrix multiplication on GPU (Naive)
__global__ void matMulNaive(int *A, int *B, int *C, int width) {
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

// Tiled matrix multiplication with shared memory
__global__ void matMulTiled(int *A, int *B, int *C, int width) {
    
    __shared__ int shared_A[16][16]; 
    __shared__ int shared_B[16][16]; 

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    for (int i = 0; i < (width / 16); i++) {
        // loading the tiles into shared memory
        shared_A[threadIdx.y][threadIdx.x] = A[row * width + (i * 16 + threadIdx.x)];
        shared_B[threadIdx.y][threadIdx.x] = B[(i * 16 + threadIdx.y) * width + col];

        __syncthreads(); // barrier sync to ensure all the tile have loaded

        // computing for current tile
        for (int j = 0; j < 16; j++) {
            sum += shared_A[threadIdx.y][j] * shared_B[j][threadIdx.x];
        }

        __syncthreads(); // ensure the completion before contiuing to the next tile
    }

    // saving the result in the global device memory
    if (row < width && col < width) {
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

    srand(time(NULL));   // Seeding the rand() with time.

    // Dynamic memory allocation
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

    // cpy from host to GPU
    cudaMemcpy(A_d, A, rows * cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, rows * cols * sizeof(int), cudaMemcpyHostToDevice);

    // defining block and grid sizes
    dim3 blockSize(16, 16, 1);  // 16 x 16 threads
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y, 1); // Grid size

    struct timer t;

    // Matrix multiplication on GPU (Naive)
    start_timer(&t);
    matMulNaive<<<gridSize, blockSize>>>(A_d, B_d, C_d, width);
    cudaDeviceSynchronize();    
    stop_timer(&t);
    printf("Matmul on GPU: %f seconds\n", time_diff(&t));

    
    cudaMemcpy(C, C_d, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

    
    start_timer(&t);
    matMulTiled<<<gridSize, blockSize>>>(A_d, B_d, C_d, width);
    cudaDeviceSynchronize();    
    stop_timer(&t);
    printf("Tiled matmul:  %f seconds\n", time_diff(&t));

    
    cudaMemcpy(C, C_d, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

    // matmul on CPU
    int *C_CPU = (int*)malloc(rows * cols * sizeof(int));  // To store results from matMul_CPU
    start_timer(&t);
    matMul_CPU(A, B, C_CPU, width);
    stop_timer(&t);
    printf("Matmul on CPU: %f seconds\n\n", time_diff(&t));

    // Verify the output
    char is_correct_naive = 1;
    char is_correct_tiled = 1;

    
    for (int i = 0; i < rows * cols; i++) {
        if (C[i] != C_CPU[i]) {
            is_correct_naive = 0;
            break;
        }
    }
    
    for (int i = 0; i < rows * cols; i++) {
        if (C[i] != C_CPU[i]) {
            is_correct_tiled = 0;
            break;
        }
    }

    if (is_correct_naive) {
        printf("Results from the GPU and the CPU match ✅\n");
    } else {
        printf("Results from the GPU and the CPU do not match ❌\n");
    }

    if (is_correct_tiled) {
        printf("Results from the GPU(tiled) and the CPU match ✅\n");
    } else {
        printf("Results from the GPU(tiled) and the CPU do not match ❌\n");
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

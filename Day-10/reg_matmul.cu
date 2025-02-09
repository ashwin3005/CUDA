#include <stdio.h>

__global__ void matMul(float *A, float *B, float *C, int width_A, int width_B, int width_C){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < width_C && col < width_C){
        float sum = 0;
        for(int i=0; i<width_A; i++){
            sum += A[row * width_A + i] + B[i * width_B + col ];
        }
        C[row * width_C + col] = sum;
    }
}


int main(){
    int A_rows = 5;
    int A_cols = 3;
    int B_rows = 3;
    int B_cols = 5;
    
    //result matrix
    int C_rows = A_rows;
    int C_cols = B_cols;

    if(A_cols != B_rows){
        printf("Not possible to do matrix multiplication (Dimension mismatch)");
        return 1;
    }

    float *A, *B, *C;
    A = (float*)malloc(A_rows * A_cols * sizeof(float));
    B = (float*)malloc(B_rows * B_cols * sizeof(float));
    C = (float*)malloc(C_rows * C_cols * sizeof(float));

    for(int i=0; i<A_rows*A_cols; i++){
        A[i]=((float)rand()/(float)(RAND_MAX)) * 20.0f - 40.0f;
    }
    for(int i=0; i<B_rows*B_cols; i++){
        B[i] = ((float)rand()/(float)(RAND_MAX)) * 20.0f - 40.0f;
    }

    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, A_rows * A_cols * sizeof(float));
    cudaMalloc((void**)&B_d, B_rows * B_cols * sizeof(float));
    cudaMalloc((void**)&C_d, C_rows * C_cols * sizeof(float));

    cudaMemcpy(A_d, A, A_rows * A_cols * sizeof(float) ,cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, B_rows * B_cols * sizeof(float) ,cudaMemcpyHostToDevice);

    // KERNEL LAUNCH
    dim3 blocksize(64, 64, 1);
    dim3 gridsize(C_cols + blocksize.x - 1/blocksize.x, C_rows + blocksize.y - 1/blocksize.y,1);

    matMul<<<gridsize, blocksize>>>(A_d, B_d, C_d, A_cols, B_cols, C_cols);

    cudaMemcpy(C, C_d, C_rows*C_cols*sizeof(float), cudaMemcpyDeviceToDevice);

    return 0;
}

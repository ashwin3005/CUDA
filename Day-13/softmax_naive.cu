/*
A simple implementation of CUDA kernel for softmax function
*/

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

/*
The below kernel parallelises the computation of softmax function.
- each thread is responsible for calculating the the softmax for a row.
- the below is a naive implementation. (we can optimize this by 
*/ 
__global__ void softMaxNaive(float *logits, float *results, int N, int K){
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N){
        float norm = 0.0f;
        float n_max = -INFINITY;
        
        // pass for calculating the maximum value in the vector.
        for(int i=0; i<K; i++){
            if(n_max<logits[row*K+i]){
                n_max = logits[row*K+i];
            }
        }
        // a pass for calculating 'norm'. i.e, summation of e^V_i
        for(int i=0; i<K; i++){
            norm += exp(logits[row*K+i]-n_max);
        }
        // a pass again for calcualting the softmax for each element.
        for(int i=0; i<K; i++){
            results[row*K+i] = exp(logits[row*K+i]-n_max)/norm;
        }
    }
}

void printMatrix(float *M, int N, int K);
struct timer t;

int main(){
    int N = 1000000;  // number of rows
    int K = 512;  // number of cols (elements per row)

    size_t bytes = N * K * sizeof(float); // num of bytes required in memory

    float *logits, *results;

    logits = (float*)malloc(bytes);
    results = (float*)malloc(bytes);

    // initialise values
    for(int i=0; i<N*K; i++){
        logits[i] = i+1;
    }

    // printf("Logits matrix: \n");
    // printMatrix(logits, N, K);

    float *logits_d, *results_d;

    cudaMalloc((void**)&logits_d, bytes);
    cudaMalloc((void**)&results_d, bytes);

    cudaMemcpy(logits_d, logits, bytes, cudaMemcpyHostToDevice);

    // kernel launch
    dim3 THREADS(32, 1, 1);
    dim3 BLOCKS((K+ THREADS.x- 1)/THREADS.x, 1, 1);

    start_timer(&t);
    softMaxNaive<<<BLOCKS, THREADS>>>(logits_d, results_d, N, K);
    cudaDeviceSynchronize();
    stop_timer(&t);

    cudaMemcpy(results, results_d, bytes, cudaMemcpyDeviceToHost);

    // printf("probablities from softmax: \n");
    // printMatrix(results, N, K);

    printf("Time taken to compute softmax: %f seconds\n", time_diff(&t));

    free(logits);
    free(results);
    cudaFree(logits_d);
    cudaFree(results_d);
    return 0;
}

void printMatrix(float *M, int N, int K){
    for(int i=0;i<N; i++){
        for(int j=0; j<K; j++){
            printf("%f ", M[i*K+j]);
        }printf("\n");
    }printf("\n");
}
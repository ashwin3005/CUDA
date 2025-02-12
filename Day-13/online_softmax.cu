/*
Online softmax 
reference: https://arxiv.org/pdf/1805.02867 (Online normalizer calculation for softmax)
*/

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

__global__ void onlineSoftMax(float *logits, float *results, int N, int K){
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < N){
    float n_max = -INFINITY;
    float norm = 0.0f;
    // pass 1
    for(int i=0; i<K; i++){
        if(n_max<logits[row*K+i]){
            // correct the global norm
            norm = norm * exp(n_max - logits[row*K+i]);
            n_max = logits[row*K+i];
        }
        norm += exp(logits[row*K+i]-n_max);
    }
    // 2nd pass
    for(int i=0; i<K; i++){
        results[row*K+i] = exp(logits[row*K+i]-n_max)/ norm;
    }
    }
}

void printMatrix(float *M, int N, int K);
struct timer t;

int main(){
    int N = 1000000;  // no. of rows
    int K = 512; // no. of col

    size_t bytes = N * K * sizeof(float);

    float *logits, *results;

    logits = (float*)malloc(bytes);
    results = (float*)malloc(bytes);

    for(int i=0; i<N*K; i++){
        logits[i] = i+1;
    }

    float *logits_d, *results_d;
    cudaMalloc((void**)&logits_d, bytes);
    cudaMalloc((void**)&results_d, bytes);

    cudaMemcpy(logits_d, logits, bytes, cudaMemcpyHostToDevice);

    // kernel launch
    dim3 THREADS(32, 1, 1);
    dim3 BLOCKS((N+THREADS.x - 1)/THREADS.x, 1, 1);

    start_timer(&t);
    onlineSoftMax<<<BLOCKS, THREADS>>>(logits_d, results_d, N, K);
    cudaDeviceSynchronize();
    stop_timer(&t);

    cudaMemcpy(results, results_d, bytes, cudaMemcpyDeviceToHost);

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
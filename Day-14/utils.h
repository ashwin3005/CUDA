#include <stdio.h>

void printMatrix(float *M, int N, int K){
    for(int i=0;i<N; i++){
        for(int j=0; j<K; j++){
            printf("%f ", M[i*K+j]);
        }printf("\n");
    }printf("\n");
}
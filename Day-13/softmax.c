/*
A simple implementation of softmax layer in C.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void softMax(double *A, double *result, int N);
void printArr(double *A, int N);

int main(){
    int N  = 10;
    size_t bytes = sizeof(double) * N;
    double *A, *result;
    A = malloc(bytes);
    result = malloc(bytes);

    // initialize values
    for(int i=0; i<N; i++){
        A[i] = i+1;
    }

    // softmax activation function
    softMax(A, result, N);

    // print the logits and the probablities
    printf("Raw outputs of logits: \n");
    printArr(A, N);
    printf("Probablities after softmax: \n");
    printArr(result, N);

    free(A);
    free(result);
    return 0;
}

void softMax(double *A, double *result, int N){
    double dinom = 0;
    for(int i=0; i<N; i++){
        dinom += exp(A[i]);
    }
    for(int i=0; i<N; i++){
        result[i] = exp(A[i])/dinom;
    }
}

void printArr(double *A, int N){
    for(int i=0; i<N; i++){
        printf("%f ", A[i]);
    }printf("\n");
}
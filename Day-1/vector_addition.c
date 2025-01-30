#include <stdio.h>
#include <stdlib.h>

void add_vector(int *A, int *B, int *result, int n){
    for(int i=0; i<n; i++){
        result[i] = A[i] + B[i];
    }
}

int main(void){
    int *A_h, *B_h, *result_h;    // declaring int pointers for arrays

    int n = 10;  // size of the array

    // Dynamic memory allocation
    A_h = malloc(n * sizeof(int));     // I am not casting the pointers :), hence it uses void ptr by default
    B_h = malloc(n * sizeof(int));
    result_h = malloc(n * sizeof(int));

    // always verify whether the malloc succeded 
    if(A_h == NULL || B_h == NULL || result_h == NULL){
        printf("Memory allocation for arrays has failed|\n");
        return 1;
    }

    // initialising A_h and B_h with placeholder values.
    for(int i=0; i<n; i++){
        A_h[i] = i;
        B_h[i] = i*2;
    }

    // function call
    add_vector(A_h, B_h, result_h, n);


    // printing the results :)
    printf("idx A + B = result\n");
    for(int i=0; i<n; i++){
        printf("%d | %d + %d = %d", i, A_h[i], B_h[i], result_h[i]);
        printf("\n");
    }


    // Don't forget
    free(A_h);
    free(B_h);
    free(result_h);

    return 0;
}
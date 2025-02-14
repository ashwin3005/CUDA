#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10       // number of elements in the array

int main(){
    int *arr;

    arr = malloc(N*sizeof(int));
    
    srand(time(NULL));   // seeding the random number with time

    for(int i=0; i<N; i++){
        arr[i] = rand()%10;  // random numbers from 0 to 9
    }

    // print the array
    for(int i=0; i<N; i++){
        printf("%d ", arr[i]);
    }printf("\n");

    return 0;
}

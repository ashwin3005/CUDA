#include <stdio.h>

__global__ void my_kernel(){
    printf("Hello world from GPU!\n");
}

int main(){
    //launch kernel
    my_kernel<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
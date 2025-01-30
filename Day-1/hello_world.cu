#include <iostream>

__global__ void hellow(){
        printf("Hello from CUDA block %d abd thread %d ! \n", blockIdx.x ,threadIdx.x);
}

int main(){
        // Launch kernel with 2 block and 4 threads
        hellow<<<2, 4>>>();

        //wait for the GPU to finish before accessing the result
        cudaDeviceSynchronize();

        return 0;
}
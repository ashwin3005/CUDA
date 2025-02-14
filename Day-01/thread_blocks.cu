#include <iostream>

// Define the kernel
__global__ void hellow(){
    printf("Hello World\nThread Id: <%d, %d, %d>\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("Block Id: <%d, %d, %d>\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("Blockdim: <%d, %d, %d>\n", blockDim.x, blockDim.y, blockDim.z);
}

int main() {
    // Launch the kernel
    hellow<<<2, 4>>>();

    cudaDeviceSynchronize();

    return 0;
}
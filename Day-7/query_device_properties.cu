#include <stdio.h>

int main(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    printf("Maximum threads per block: %d\n", devProp.maxThreadsPerBlock);
    printf("Number of SM in this device: %d\n",devProp.multiProcessorCount);
    printf("Clock frequency of this device: %d\n", devProp.clockRate);
    printf("Warp size: %d\n", devProp.warpSize);
    printf("Maximum number of registers per block: %d\n", devProp.regsPerBlock);
}
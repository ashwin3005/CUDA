# Chapter 02 - Heterogeneous data parallel computing

We saw about the design philosophies of CPUs and GPUs and how they are different from each other in the last chapter. 

Now, we call a system heterogeneous if it has both a CPU and GPU in it.

In a heterogeneous system, generally we call the 
- CPU as "host", and 
- GPU as "device"


## Data parallelism

When modern software applications run slowly, the problem is usually data—too much data to process.

In several use-cases the data-point in those applications as independent of each other and we can process them in parallel.

Example:
- Pixels of an images (we converting from RGB to grey scale)
- billions of grid points while similating a fluid dynamics model etc..

< we can list even more examples>

Such independent evaluation of different pieces of data is the basis of **data parallelism**.


## Writing code in CUDA C

The structure of a CUDA C program reflects the coexistence of a host (CPU)
and one or more devices (GPUs) in the computer. 

Each CUDA C source file can have a mixture of host code and device code. By default, **any traditional C program is a CUDA program that contains only host code.** 

we can add device code into any C source file. The device code is clearly marked with special CUDA C keywords. 

The point to note here is " all the C code is a valid CUDA C code "

### Compiling C code with NVCC 
file name: *hello.cu*
```C
#include <stdio.h>

int main(void){
    printf("Hello world! (compiled with nvcc)");
    return 0;
}
```
Compiling 
```bash
nvcc hello.cu -o hello
```
Executing
```bash
./hello
```

### Hello World in CUDA C
```C
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
```

### Explanation

- A **kernel** is a function that runs on the GPU (device) and can be called from the CPU (host).

- **\_\_global__** is a keyword used to define a function as a kernel.

- When we call a kernel, we typically mention it as "launcing a kernel"

- When we launch a kernel, we specify the number of Blocks and Threads inside angular brackets (" <<<Blocks, Threads>>> ")

- A Grid has multiple blocks, A Block has multiple Thread.

The parallel portion of our applications is executed K times in parallel by K different CUDA threads, as opposed to only one time like regular C/C++ functions. 

This explained more clearly on this blog by NVIDIA [CUDA programming model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model)

I have simply printed out the values of threadId.x, blockId.x [here](./thread_blocks.cu)


## Vector Addition

#### In C
First I have implemented a simple vector addition in C.

code: [Vector Addition in C](./vector_addition.c)

#### In cuda

I've later implemented a the same in CUDA C :) 
code: [Vector Addition in CUDA C](./vector_addition.cu)
# Chapter 02 - Heterogeneous Parallel Computing

Today I am continuing on chapter 02. And building on top of what I have learnt yesterday.

When we launch a CUDA kernel, since all the threads execute the same code, CUDA C programming is an instance of the well-
known single-program multiple-data (SPMD) parallel programming style.

When we launch a kernel, the CUDA runtime system launches a **grid of threads** that are organized into a two-level hierarchy. 
- Each grid is organized as an array of thread blocks, or blocks in general.
- All blocks of a grid are of the same size. 
- Each block can contain up to **1024 threads** on current systems.

## Built-in Variables

- For a given grid of threads, the number of threads in a block is available in a built-in variable named **blockDim**.

- The blockDim variable is a struct of three unsigned integer (x, y, z).


We have two other built-in variable when a kernel is launched, **threadIdx** and **blockIdx**. (they both are structs of three unsigned integers (x, y, z))

The threadIdx is unique inside a block and similarly the blockIdx is unique within a grid.
they start with 0 and increase by 1.

eg: 0, 1, 2, 3.. N-1.


Each thread can combine its threadIdx and blockIdx values to create a unique global index for itself within
the entire grid.

$$ i=blockIdx.x * blockDim.
x + threadIdx.x $$

Tip 💡 : *In general, it is recommended that the number of threads in each dimension of a thread block be a multiple of 32 for hardware efficiency reasons.*

## Qualifier Keywords

CUDA C provides three qualifier keywords that can be used in function declarations.

- \_\_global__
- \_\_host__
- \_\_device__


**\_\_global__** kernel function is executed on the
device and can be called from the host.

**\_\_host__** fuction is executed on host(CPU) and can only be called from the host. (like a regular C function)

**\_\_device__** keyword indicates that the function is declared on the GPU(device) and can only be called from a kernel function or another device function. 

The device function is executed by the device thread that calls it and does not result in any new
device threads being launched


- By default all the function are \_\_host__ functions.

## Compilation

The NVCC compiler processes a CUDA C program, using the CUDA keywords to separate the host code and device code. 

The host code is straight ANSI C code, which is compiled with the host’s standard C/C++ compilers and is run as a traditional CPU process. 

The device code, which is marked with CUDA keywords that designate CUDA kernels and their associated helper functions and data structures, is compiled by NVCC into virtual binary files called **PTX** files. 

These PTX files are further compiled by a runtime component of NVCC into the real object files and
executed on a CUDA-capable GPU device.

---

*I have started enjoying GPU programming! ❤️*
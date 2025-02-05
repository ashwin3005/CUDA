# Chapter 04 - Compute Architecture and Scheduling

In the previous chapters we have seen,
- what is a GPU?, 
- how it is different from a CPU?.
- What are kernel, threads, blocks and grids?,
- How to launch a kernel? etc..

In the chapter 4, 5, and 6 we learn about compute architecture, memory architecture and performance optimization respectively.

## GPU Architecture Overview (4.1):
- Modern CUDA-capable GPUs are designed with an array of streaming multiprocessors (SMs), each consisting of multiple cores (CUDA cores). 

- These cores share control logic and memory resources. 

- For example, the NVIDIA Ampere A100 has 108 SMs, each with 64 cores, totaling 6912 cores in the entire GPU.

## Block Scheduling (4.2):
- When a kernel is launched, it creates a grid of threads, which are distributed among the SMs on a **block-by-block** basis. 

- Each block is scheduled onto a single SM, but multiple blocks can be assigned to the same SM.

- The number of blocks that can be assigned to a single SM is limited by the available hardware resources. 

- The CUDA runtime system manages this by assigning new blocks to SMs as others complete their execution.

## Synchronization and Scalability (4.3):
**Barrier synchronization:** Threads within a block can synchronize their execution using __syncthreads(). This ensures that all threads in a block reach a specific point in their execution before proceeding. 

- It's like people waiting for each other at a barrier before continuing their activities. (In the book they have illustrated this with "friends shopping in a mall" example)

- CUDA allows **transparent scalability**, meaning that the same program can run on GPUs with different numbers of SMs, adjusting the execution according to the hardware resources available.

- Note: we must be carefull while using __syncthreads(), we must ensure all the thread in a block passes thorough same 'barrier sync', else this might lead to **deadlocks.**  (Avoid using this in side if-else in general.)

## Warps and SIMD Hardware (4.4):
- A block is divided into warps, which are groups of **32 threads**. The SMs execute threads in a warp following the Single Instruction, Multiple Data (SIMD) model. i.e, all threads in a warp execute the same instruction simultaneously on different data elements.

- Warps are scheduled on processing blocks within the SM, and each processing block shares an instruction fetch/dispatch unit that manages the execution of all threads in the warp.

- Grouping the threads into warps is straightforward in 1-D threads, for multidimentional threads first the hardware will project them into a linear one and then group them into warps.

- If the number of threads are not divisible by 32, the last warp will be padded with extra inactive threads

---
I have implemented a simple ReLU kernel in CUDA C

- CUDA C implementation: [ReLU](./Relu.cu)

## ReLU (Rectified Linear Unit)

ReLU is a type of activation function that are linear in the positive dimension, but zero in the negative dimension.

refer: [ReLU 'Papers with code'](https://paperswithcode.com/method/relu)

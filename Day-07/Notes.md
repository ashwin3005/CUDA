# Chapter 04 - Compute Architecture and Scheduling (continuation)

## Control Divergence

- Threads within a warp execute the same instruction simultaneously (SIMD). **Control divergence** occurs when threads in a warp follow different execution paths (e.g., some take the `if` path, others the `else` path).
- When divergence occurs, the hardware executes each path in separate passes, with threads inactive during paths they don’t follow. This preserves thread independence but incurs performance overhead.
- **Examples**:
  - **If-Else Divergence**: In a warp of 32 threads, if 24 take the `if` path and 8 take the `else` path, the hardware executes two passes: one for the `if` path and one for the `else` path.
  - **For-Loop Divergence**: In a loop, threads may complete iterations at different times, causing divergence in later iterations.

- Performance impact decreases with larger datasets. 
For example, if we perform a vector addition on a vector of size 1000,
we may allocate 1024 threads. this means we will use 32 warps. in this only 8 threads will be active in the last warp.

- In the above example, only 1 out of 32 warps may diverge, resulting in a minimal performance impact (~3%).

#### 4.6 Warp Scheduling and Latency Tolerance
- **Warp Scheduling**: GPUs assign more number of threads for SMs than the cores it has. When a warp stalls (e.g., waiting for memory access), the SM switches to another ready warp. This is ofter called “latency tolerance” or “latency hiding"
- **Anology in the PMPP book**: Similar to a post office clerk serving other customers while one fills out a form, GPUs execute other warps while one waits for long-latency operations.


#### 4.7 Resource Partitioning and Occupancy
- **Dynamic Resource Partitioning**: SMs dynamically allocate resources (registers, shared memory, thread slots) to threads and blocks. This flexibility allows SMs to handle varying block sizes and resource needs.
- **Occupancy**: The ratio of active warps to the maximum supported warps. High occupancy improves latency tolerance but can be limited by resource constraints (e.g., registers, shared memory).
- **Examples**:
In the book they have given multiple examples of hardware limitations.
  
  - The example given is the Ampere A100 GPU. It can have up to 2048 threads per SM (which is 64 warps, since each warp has 32 threads). If a block has 1024 threads, each SM can hold 2 blocks (2*1024=2048). If blocks are smaller, like 512 threads, you can fit 4 blocks (4*512=2048). 
  
  - But there's also a limit on the number of blocks per SM. For instance, if each block has 32 threads, you'd need 64 blocks to reach 2048 threads, but the SM might only allow 32 blocks. So you end up with 32*32=1024 threads, leading to 50% occupancy.

  - Another example is block sizes that don't divide evenly into the maximum threads. A block size of 768 would let 2 blocks run (2*768=1536), leaving 512 threads unused, resulting in 75% occupancy.

  - Then there's the impact of registers. Each thread uses some registers, and the total per SM is limited. For example, A100 has 65,536 registers. If each thread uses 32 registers, 2048 threads would need 32*2048=65,536, which is perfect. But if each thread uses 64 registers, only 1024 threads can fit (65,536/64=1024), cutting occupancy to 50%. Adding even a couple more registers per thread (from 31 to 33) could push the total over the limit, forcing fewer blocks and lower occupancy, which is a **performance cliff.**

*The above examples was awesome  ❤️, and it was fasinating to read it for the first time. it made me clear how a GPU is scheduling the threads!*

#### 4.8 Querying Device Properties
- **Device Properties**: CUDA provides APIs to query device capabilities (e.g., number of SMs, max threads per block, warp size, registers per block). This allows applications to adapt to different hardware.
- **Examples**:
  - **cudaGetDeviceCount**: Returns the number of CUDA devices in the system.
  - **cudaGetDeviceProperties**: Retrieves properties like `maxThreadsPerBlock`, `multiProcessorCount`, and `regsPerBlock`.
- **Use Cases**: Applications can use device properties to optimize kernel launch configurations (e.g., block size, grid size) and ensure compatibility across hardware generations.

# Chapter 05 - Memory Architecture and Data locality

## Importance of memory access efficiency
- Memory access efficiency is crucial for achieving high performance in CUDA kernels.
- 'device global memory' (off-chip DRAM) has high latency and limited bandwidth, which can limit the performance of the kernel.

#### Compute to Global Memory Access Ratio:

- The **compute to global memory access ratio** (FLOP/B) is a key metric that indicates how efficiently the kernel uses memory for each floating-point operation.

- This is also called arithmetic intensity or computional intensity.

- A low ratio indicates that performance is constrained by memory bandwidth, rather than by computation. 

- Example, in our matrix multiplication we perform dot products, in which we do a floating-point multiplication and then we do floating-point addition to accumulate them in the sum variable. i.e, we perform 2 floating-point operations (FLOPs) per 8 byte (2 X 4 bytes). and hence the compute to global memory access ratio is 0.25 FLOP/B.

---

In general, If a kernel is limited by memory bandwidth, it’s termed as **memory-bound**. In this case, improving memory access efficiency is crucial for better performance.

#### The Roofline Model:

The [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model) helps visualize performance limits:
The x-axis shows the computational intensity (FLOP/B).
The y-axis shows computational throughput (GFLOPS).
A horizontal line represents peak computational throughput, and a sloped line represents peak memory bandwidth.
Applications with low computational intensity are memory-bound, while those with high intensity are compute-bound.

also refer: [Slides from BERKELEY PAR LAB](https://crd.lbl.gov/assets/pubs_presos/hotchips08-roofline-talk.pdf)

## Resources Used
- [DRAM vs SRAM](https://forums.developer.nvidia.com/t/gpu-memory-dram-or-sram/26742) - This discussion explains difference between "device Global Memory" and "on-chip memory".
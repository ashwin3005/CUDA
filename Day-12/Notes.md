# Chapter 06 - Performance considerations

## Memory Coalescing
Global Memory Access in CUDA:

Global memory bandwidth is a critical factor in CUDA kernel performance.

CUDA applications process large amounts of data from global memory, making efficient access patterns essential.

### DRAM Basics:

- DRAM cells store data as electrical charges in capacitors.

- Accessing DRAM is slow (tens of nanoseconds) compared to modern processor speeds (sub-nanosecond cycles).

- DRAM uses parallelism (bursts) to improve data access throughput.

### Memory Coalescing:

- Coalescing combines multiple memory accesses into a single request when threads in a warp access consecutive memory locations.

- Favorable access patterns: Threads in a warp access consecutive global memory locations (e.g., thread 0 accesses location X, thread 1 accesses X+1, etc.).

- Unfavorable patterns: Threads access non-consecutive locations, leading to uncoalesced accesses and reduced performance.

### Row-Major vs. Column-Major Order:

- In row-major order, consecutive threads accessing consecutive elements in a row leads to coalesced accesses.

- In column-major order, consecutive threads access elements far apart in memory, causing uncoalesced accesses.

### Optimization Strategies:

- Rearrange thread-to-data mapping or data layout to improve coalescing.

- Use shared memory to handle unfavorable access patterns after loading data in a coalesced manner.

- Example: Corner turning optimizes access patterns for column-major matrices by reordering thread assignments.

### Analogy to Traffic Congestion:

- Memory coalescing is like carpooling: combining multiple data accesses into one request reduces traffic (memory bandwidth usage).

- Threads in a warp are ideal for coalescing because they execute simultaneously (SIMD).

## Hiding Memory Latency
### DRAM Parallel Organization:

- DRAM systems use channels and banks to increase access bandwidth.

- Each channel connects multiple banks to the processor via a bus.

- Data transfer bandwidth depends on bus width and clock frequency (e.g., DDR buses transfer data on both rising and falling clock edges).

### Banking for Latency Hiding:

- DRAM access latency (time to read/write cells) is much longer than data transfer time.

- Multiple banks allow overlapping access latencies: while one bank is transferring data, another can start accessing its cells.

- To fully utilize bus bandwidth, the number of banks should exceed the ratio of access latency to data transfer time.

### Bank Conflicts:

- Bank conflicts occur when multiple accesses target the same bank, reducing parallelism.

- More banks reduce the likelihood of conflicts and improve performance.

### Interleaved Data Distribution:

- Data is distributed across banks and channels to ensure even access patterns.

- Example: Array elements are spread across channels and banks in an interleaved manner to maximize parallelism.

### Thread Execution and Memory Parallelism:

- High thread occupancy ensures enough memory requests to hide DRAM latency.

- Parallel thread execution and DRAM organization work together: threads access data from different banks and channels simultaneously, improving throughput.

- Caches in modern GPUs help combine redundant accesses, reducing DRAM traffic.

### Example: Matrix Multiplication:

- Thread blocks load tiles of data in parallel, with coalesced accesses distributed across channels and banks.

- Caches combine redundant accesses from different thread blocks, further optimizing performance.
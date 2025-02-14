# Chapter 01 - Introduction 

This chapter explains a history of computing. 

In 1980s and 1990s we saw a rise in the development of single-core CPUs. They got better at increasing clock frequency and hardware
resources.

single-CPU microprocessors brought GFLOPS, or giga ($10^9$) floating-point operations per second, to the desktop and TFLOPS, or tera ($10^12$) floating-point operations per second, to data centers.

However, this drive has slowed down since 2003, owing to energy consumption
and heat dissipation issues.

Then we saw a rise in CPUs with mulitple cores.

From there on the semiconductor industry has settled on two main trajectories for designing microprocessors: 
- **multi-core CPUs**, optimized for sequential performance 
- **many-thread GPUs**, optimized for parallel throughput. 

GPUs are designed for tasks like graphics rendering, focus on high throughput and can handle large numbers of threads simultaneously. This makes them ideal for parallel applications like deep learning, while CPUs remain better suited for single-threaded tasks.

To sum-up,
- CPUs are latency-oriented design
- GPUs are throughput-oriented design

## Note: 
For programs that have one or very few threads, CPUs
with lower operation latencies can achieve much higher performance than GPUs.

When a program has a large number of threads, GPUs with higher execution
throughput can achieve much higher performance than CPUs.
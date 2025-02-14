# Chapter 05 - Memory Architecture and Data locality (continuation)

## CUDA memory types

- global memory, 
- constant memory, 
- local memory, 
- registers, 
- shared memory

---
- **Global Memory (VRAM)**: Largest capacity, relatively slower access.
- **Shared Memory**: Small, fast on-chip memory used by threads in a block.
- **Constant Memory**: Fast, read-only memory for constants.
- **Registers**: Extremely fast, per-thread storage.
- **Local Memory**: Slower than registers, used for thread-specific data overflow.

## Tiling for reduced memory traffic

code : [tiled_matmul.cu](./tiled_matmul.cu)

# Chapter 03 - Multidimensional grids and data

## Multidimensional grid organisation

Execution configuration parameters in a kernel call statement specify the dimensions of the grid and the dimensions of each block.

i.e, gridDim and blockDim (both are built-in variables)

example:
```C
exampleKernel<<<gridDim, blockDim>>>(...);
```

The values that we pass inside '<<<>>>' are known as 'execution configuration parameters'. They are of type **dim3**.
- dim3 is an integer vector type of three elements x, y, and z.

for example, let's call a kernel
```C
dim3 grid_dim(4, 1, 1);
dim3 block_dim(32, 1, 1);

// launch a kernel
kernel<<<grid_dim, block_dim>>>(...);
```

The above code will launch a kernel with 5 blocks and 32 thread each. hence the total number of threads would be 5 X 32 = 160.

## blockDim

As we all know, the total no. of threads per block must not exceed 1024. We can distribute them across x,y,z however we want as long as the former is true.

example: 
- Allowed: (516, 1, 1), (8, 16, 4), (32, 16, 2)
- Not allowed: (32, 32, 2)


A grid and its blocks do not need to have the same dimensionality. A grid can
have higher dimensionality than its blocks and vice versa.


## Note
If we have a 2 dimensional grid of blocks. let's say 2 X 2, the block (1, 0) has blockIdx.y = 1 and blockIdx.x = 0.

Note that the ordering of the block and thread labels is such that **highest dimension comes first**.

For example, thread (1,0,2) has threadIdx.z = 1,
threadIdx.y = 0, and threadIdx.x = 2.
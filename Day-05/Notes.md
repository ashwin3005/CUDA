# Chapter 03 - Multidimensional grids and data

I am continuing on chapter three. (I have started this on Day-3)


CUDA supports multidimensional grids (Up to 3D)
- This simplifies processing multidimensional data.
- If we did not have multidimensional grid support, we must convert them into single-dimensionals indices and then process them in parallel.


## Indexing in multidimensional grids

### For 2D
```C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
```


#### **Note:** although we are using the 2 Dimensional data, it is strored in the DRAM(global device memory) in a linear manner (we use row major)

---
While writing parallel programs be careful on **indexing** and **boundry checks**

I have refered [this](https://www.youtube.com/watch?v=ovAB0jf8QGg) lecture to understand matmul.
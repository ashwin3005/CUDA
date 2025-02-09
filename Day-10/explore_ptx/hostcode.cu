#include <stdio.h>
#include <cuda.h>

int main() {
    // Initialize CUDA
    cuInit(0);

    // Create a CUDA context
    CUcontext context;
    cuCtxCreate(&context, 0, 0);

    // Load the PTX file with printf support
    CUmodule module;
    CUjit_option options[] = { CU_JIT_LOG_VERBOSE };
    void* optionValues[] = { (void*)1 }; // Enable verbose logging (includes printf support)
    cuModuleLoadDataEx(&module, "hello.ptx", 1, options, optionValues);

    // Get the kernel function from the PTX module
    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, "my_kernel");

    // Launch the kernel
    void *args[] = {};  // No arguments for this kernel
    cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);

    // Synchronize to ensure the kernel completes
    cuCtxSynchronize();

    // Clean up
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}
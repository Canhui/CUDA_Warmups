#include <stdio.h>

int main()
{
    cudaDeviceProp prop;
    int count;
    int driver_version;
    int runtime_version;

    cudaGetDeviceCount(&count);

    for(int i = 0; i < count; i++)
    {
        cudaGetDeviceProperties(&prop, i);
        cudaDriverGetVersion(&driver_version);
        cudaRuntimeGetVersion(&runtime_version);

        printf("\n------------------------Device ID: %d (general info)------------------------\n", i);
        printf("Device name: %s\n", prop.name);
        printf("Driver version: %d.%d\n", driver_version/1000, (driver_version%100)/10);
        printf("Runtime version: %d.%d\n", runtime_version/1000, (runtime_version%100)/10);
        printf("Compute capability version: %d.%d\n", prop.major, prop.minor);
        printf("Clock rate: %.0f MHz\n", prop.clockRate * 1e-3f);

        printf("Concurrent kernels: %s\n", prop.concurrentKernels? "Yes":"No");
        #if CUDART_VERSION >= 5000
            printf("Concurrent copy and kernel execution: %s, with: %d copy engines\n", (prop.deviceOverlap? "Yes":"No"), prop.asyncEngineCount);
        #endif

        printf("Kernel execution timeout: %s\n", prop.kernelExecTimeoutEnabled? "Yes":"No");
        printf("Integrated GPU sharing host memory: %s\n", prop.integrated? "Yes":"No");
        printf("Support host page locked memory mapping: %s\n", prop.canMapHostMemory? "Yes":"No");

        printf("\n------------------------Device ID: %d (memory info)-------------------------\n", i);
        #if CUDART_VERSION >= 5000
            printf("Memory clock rate: %f Mhz\n", prop.memoryClockRate*10e-7);
            printf("Memory bus width: %d-bit\n", prop.memoryBusWidth);
        #endif

        printf("Total global memory: %lf Mbytes\n", prop.totalGlobalMem/1048576.0);
        printf("Total constant memory: %ld bytes\n", prop.totalConstMem);
        printf("Max memory pitch: %ld bytes\n", prop.memPitch);

        printf("\n------------------------Device ID: %d (MP info)-----------------------------\n", i);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("Shared memory per block: %ld bytes\n", prop.sharedMemPerBlock);
        printf("Registers per block: %d\n", prop.regsPerBlock);
        printf("Threads per warp: %d\n", prop.warpSize);

        #if CUDART_VERSION >= 5000
            printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        #endif

        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    }

    return 1;
}

---------------------

本文来自 林微 的CSDN 博客 ，全文地址请点击：https://blog.csdn.net/Canhui_WANG/article/details/82668060?utm_source=copy 
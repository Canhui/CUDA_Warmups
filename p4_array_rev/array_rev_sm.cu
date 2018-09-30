#include <stdio.h>

__global__ void array_reverse(int *array_a_dev, int *array_a_rev_dev, int len)
{
    int tid = threadIdx.x;
    array_a_rev_dev[len - tid - 1] = array_a_dev[tid];
}

__global__ void array_reverse_shared(int *array_a_dev, int *array_a_rev_dev, int len)
{
    int tid = threadIdx.x;
    __shared__ int array_shared[9];
    array_shared[tid] = array_a_dev[tid];
    __syncthreads();
    array_a_rev_dev[len - tid - 1] = array_shared[tid];
}

__global__ void array_reverse_dynamic_shared(int *array_a_dev, int *array_a_rev_dev, int len)
{
    int tid = threadIdx.x;
    extern __shared__ int array_shared[];
    // __shared__ int array_shared[9];
    array_shared[tid] = array_a_dev[tid];
    __syncthreads();
    array_a_rev_dev[len - tid - 1] = array_shared[tid];
}

int main()
{
    int len = 9;

    int *array_a_host;
    int *array_a_rev_host;
    int *array_result_back_to_host;

    array_a_host = (int *)malloc(len * sizeof(int));
    array_a_rev_host = (int *)malloc(len * sizeof(int));
    array_result_back_to_host = (int *)malloc(len * sizeof(int));

    for(int i = 0; i < len; i++)
    {
        array_a_host[i] = i * 2;
    }

    for(int i = 0; i < len; i++)
    {
        array_a_rev_host[len - i - 1] = array_a_host[i];
    }

    printf("\n-------------Array a-----------------\n");
    for(int i = 0; i < len; i++)
    {
        printf("%d ", *(array_a_host + i));
    }
    printf("\n");

    printf("\n-------------Array b-----------------\n");
    for(int i = 0; i < len; i++)
    {
        printf("%d ", *(array_a_rev_host + i));
    }
    printf("\n");


    // ------------------GPU--------------------------
    int *array_a_dev;
    int *array_a_rev_dev;

    cudaMalloc((void**) &array_a_dev, len * sizeof(int));
    cudaMalloc((void**) &array_a_rev_dev, len * sizeof(int));

    cudaMemcpy(array_a_dev, array_a_host, len * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(9, 1, 1);
    // Version 1
    // array_reverse<<<dimGrid, dimBlock>>>(array_a_dev, array_a_rev_dev, len);
    // Version 2
    array_reverse_shared<<<dimGrid, dimBlock>>>(array_a_dev, array_a_rev_dev, len);
    // Version 3
    // array_reverse_dynamic_shared<<<dimGrid, dimBlock, len*sizeof(int)>>>(array_a_dev, array_a_rev_dev, len);

    cudaMemcpy(array_result_back_to_host, array_a_rev_dev, len * sizeof(int), cudaMemcpyDeviceToHost);  

    printf("\n-------------Array cuda--------------\n");
    for(int i = 0; i < len; i++)
    {
        printf("%d ", *(array_result_back_to_host + i));
    }
    printf("\n");


    free(array_a_host);
    free(array_a_rev_host);
    free(array_result_back_to_host);

    cudaFree(array_a_dev);
    cudaFree(array_a_rev_dev);

    return 1;
}
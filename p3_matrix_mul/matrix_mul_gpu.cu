#include <stdio.h>

__global__ void matrixs_1D_multiplication(int *matrix_a_dev, int *matrix_b_dev, int *matrix_c_dev, int matrix_width)
{
    int row = threadIdx.y;
    int col = threadIdx.x;

    if(row < matrix_width && col < matrix_width)
    {
        for(int k = 0; k < matrix_width; k++)
        {
            matrix_c_dev[row * matrix_width + col] += matrix_a_dev[row * matrix_width + k] * matrix_b_dev[k * matrix_width + col]; 
        }
    }
}


int main()
{
    int matrix_width = 3;

    int *matrix_a_host;
    int *matrix_b_host;
    int *matrix_c_host;

    matrix_a_host = (int *)malloc(matrix_width*matrix_width*sizeof(int));
    matrix_b_host = (int *)malloc(matrix_width*matrix_width*sizeof(int));
    matrix_c_host = (int *)malloc(matrix_width*matrix_width*sizeof(int));

    for(int row = 0; row < matrix_width; row++)
    {
        for(int col = 0; col < matrix_width; col++)
        {
            matrix_a_host[row * matrix_width + col] = row + col;
            matrix_b_host[row * matrix_width + col] = row * col + col;
        }
    }

    // ------------------GPU--------------------------
    int *matrix_a_dev;
    int *matrix_b_dev;
    int *matrix_c_dev;

    cudaMalloc((void**) &matrix_a_dev, matrix_width*matrix_width*sizeof(int));
    cudaMalloc((void**) &matrix_b_dev, matrix_width*matrix_width*sizeof(int));
    cudaMalloc((void**) &matrix_c_dev, matrix_width*matrix_width*sizeof(int));

    cudaMemcpy(matrix_a_dev, matrix_a_host, matrix_width*matrix_width*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_b_dev, matrix_b_host, matrix_width*matrix_width*sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(3, 3, 1);
    matrixs_1D_multiplication<<<dimGrid, dimBlock>>>(matrix_a_dev, matrix_b_dev, matrix_c_dev, matrix_width);

    cudaMemcpy(matrix_c_host, matrix_c_dev, matrix_width*matrix_width*sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n-------------Matrix c-----------------\n");
    for(int i = 0; i < matrix_width * matrix_width; i++)
    {
        if((i + 1) % matrix_width)
            printf("%d ", *(matrix_c_host + i));
        else
            printf("%d \n", *(matrix_c_host + i));
    }

    free(matrix_a_host);
    free(matrix_b_host);
    free(matrix_c_host);
    cudaFree(matrix_a_dev);
    cudaFree(matrix_b_dev);
    cudaFree(matrix_c_dev);

    return 1;
}
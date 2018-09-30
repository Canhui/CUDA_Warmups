#include <stdio.h>
#include <math.h>

__global__ void add_in_parallel(int *array_a, int *array_b, int *array_c)
{
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   array_c[tid] = array_a[tid] + array_b[tid];
}

int main()
{
   // --------------------------------------------
   printf("Begin\n");
   int arraysize = 1000;
   int *a_host;
   int *b_host;
   int *c_host;
   int *devresult_host;

   a_host = (int *)malloc(arraysize*sizeof(int));
   b_host = (int *)malloc(arraysize*sizeof(int));
   c_host = (int *)malloc(arraysize*sizeof(int));
   devresult_host = (int *)malloc(arraysize*sizeof(int));

   for (int i = 0; i < arraysize; i++)
   {
      a_host[i] = i;
      b_host[i] = i;
      c_host[i] = a_host[i] + b_host[i];
   }

   // ---------------------------------------------
   printf("Allocating device memory\n");
   int *a_dev;
   int *b_dev;
   int *c_dev;

   cudaMalloc((void**) &a_dev, arraysize*sizeof(int));
   cudaMalloc((void**) &b_dev, arraysize*sizeof(int));
   cudaMalloc((void**) &c_dev, arraysize*sizeof(int));

   // ----------------------------------------------
   printf("Copy host data to device\n");
   cudaMemcpy(a_dev, a_host, arraysize*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(b_dev, b_host, arraysize*sizeof(int), cudaMemcpyHostToDevice);

   // ----------------------------------------------
   printf("Add in parallel\n");
   int blocksize = 512;
   int blocknum = ceil(arraysize/double(blocksize));
   
   dim3 dimBlock(blocksize, 1, 1);
   dim3 dimGrid(blocknum, 1, 1);
   
   add_in_parallel<<<dimGrid, dimBlock>>>(a_dev, b_dev, c_dev);
   cudaThreadSynchronize();


   // ----------------------------------------------
   printf("Copy dev data to host\n");
   cudaMemcpy(devresult_host, c_dev, arraysize*sizeof(int), cudaMemcpyDeviceToHost);

   // -------------------------------------------------
   printf("Verify result\n");
   int status = 0;
   for (int i = 0; i < arraysize; i++)
   {
      // printf("%d ", devresult_host[i]);
      // printf("%d ", c_host[i]);
      if (c_host[i]!=devresult_host[i])
      {
         status = 1;
      }
   }

   if (status)
   {
      printf("Failed vervified.\n");
   }
   else
   {
      printf("Sucessdully verified.\n");
   }

   // ----------------------------------------------
   printf("Free dev memory\n");
   cudaFree(a_dev);
   cudaFree(b_dev);
   cudaFree(c_dev);

   // ----------------------------------------
   printf("Free host memory\n");
   free(a_host);
   free(b_host);
   free(c_host);

   return 1;
}
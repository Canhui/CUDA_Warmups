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
   printf("Begin...\n");
   int arraysize = 100000;
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
   }

   // ---------------------------------------------
   printf("Allocating device memory...\n");
   int *a_dev;
   int *b_dev;
   int *c_dev;

   cudaMalloc((void**) &a_dev, arraysize*sizeof(int));
   cudaMalloc((void**) &b_dev, arraysize*sizeof(int));
   cudaMalloc((void**) &c_dev, arraysize*sizeof(int));

   // ----------------------------------------------
   cudaEvent_t start,stop;
   float time_from_host_to_dev;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);
   cudaMemcpy(a_dev, a_host, arraysize*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(b_dev, b_host, arraysize*sizeof(int), cudaMemcpyHostToDevice);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(start);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time_from_host_to_dev, start, stop);
   printf("Copy host data to device, time used: %0.5g seconds\n", time_from_host_to_dev/1000);

   // ----------------------------------------------
   float time_of_kernel;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);
   int blocksize = 512;
   int blocknum = ceil(arraysize/double(blocksize));

   dim3 dimBlock(blocksize, 1, 1);
   dim3 dimGrid(blocknum, 1, 1);
   
   add_in_parallel<<<dimGrid, dimBlock>>>(a_dev, b_dev, c_dev);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(start);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time_of_kernel, start, stop);
   printf("Add in parallel, time used: %0.5g seconds\n", time_of_kernel/1000);


   // ----------------------------------------------
   float time_from_dev_to_host;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);
   cudaMemcpy(devresult_host, c_dev, arraysize*sizeof(int), cudaMemcpyDeviceToHost);
   cudaEventRecord(stop, 0);
   cudaEventSynchronize(start);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time_from_dev_to_host, start, stop);
   printf("Copy dev data to host, time used: %0.5g seconds\n", time_from_dev_to_host/1000);

   // -------------------------------------------------
   printf("Verify result...\n");
   int status = 0;
   clock_t start_cpu, end_cpu;
   float time_cpu;
   start_cpu = clock();
   for (int i = 0; i < arraysize; i++)
   {
      c_host[i] = a_host[i] + b_host[i];
   }
   end_cpu = clock();
   time_cpu = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

   for (int i = 0; i < arraysize; i++)
   {
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

   // ----------------------------------------
   printf("\nPerformance: CPU vs. GPU\n");
   printf("time cpu:%f\n",  time_cpu);
   printf("time gpu(kernel):%f\n",  time_of_kernel/1000);

   return 1;
}
#ifndef KERNEL_H_
#define KERNEL_H_

#define THREADS 64
#define BLOCKS 64

__device__ float operation_run_cuda(int operation, float a, float b);
__device__ double logloss_cuda(double actual, double predicted);
__device__ float eval_rpn_cuda(node *rpn, float *features);
__device__ float tournament(node *rpn, float *dataset, int dataset_size);
__global__ void run_cuda(node **population, float **features, int features_count, float *results);

#endif

#include "node.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h>

#ifndef KERNEL_H_
#define KERNEL_H_

#define THREADS 32 
#define BLOCKS 16 
#define FEATURE_COUNT 50
#define DATASET_SIZE 40000

#ifdef __cplusplus
  extern "C"
#endif

void prepare_and_run_cuda(node *population, float *d_features, int features_count, float *results, int pop_size, float *results_cuda);

__device__ float operation_run_cuda(int operation, float a, float b);
__device__ char operation_label_cuda(int operation);

__device__ double logloss_cuda(double actual, double predicted);
__device__ float eval_rpn_cuda(node *rpn, float *features);
__device__ float tournament_cuda(node *rpn, float *dataset, int dataset_size);
__device__ void display_rpn_cuda(node *rpn);
__device__ void display_feature_line_cuda(float *line, int feature_count);
__global__ void run_cuda(node *population, float *features, int features_count, float *results);

#endif

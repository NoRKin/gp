/*#include "kernel.cuh"*/

__device__ float operation_run_cuda(int operation, float a, float b) {
  if (operation == 0) {
    return __fadd_rn(a, b);
  }
  if (operation == 1) {
    return __fmul_rn(a, b);
  }
  if (operation == 2) {
    return __fdiv_rn(a, b);
  }
  if (operation == 3) {
    return __fsub_rn(a, b);
  }
  if (operation == 4) {
    return __fsqrt_rn(a);
  }
  if (operation == 5) {
    return __tanf(a);
  }
  if (operation == 6) {
    return __logf(a);
  }
  if (operation == 7) {
    return __expf(a);
  }
  if (operation == 8) {
    return __cosf(a);
  }
  if (operation == 9) {
    return __sinf(a);
  }

  return 0;
}

__device__ double logloss_cuda(double actual, double predicted) {
  double eps = 1e-15;

  if (predicted < eps)
    predicted = eps;

  if (predicted == 1)
    predicted -= eps;

  return - (actual * __logf(predicted) + (1 - actual) * __logf(1 - predicted));
}

__device__ float eval_rpn_cuda(node *rpn, float *features) {
  float stack[16];
  int s_index = 0;
  int i = 0;

  while(rpn[i].operation != -2) {
    if (rpn[i].operation == -1) {
      stack[s_index] = features[rpn[i].feature];
      s_index++;
    }
    // Function, we want to reduce the stack
    else if (s_index > 1) {
      s_index = s_index - 1;
      stack[s_index - 1] = operation_run_cuda(rpn[i].operation,
                                         stack[s_index - 1],
                                         stack[s_index]);
    }
    i++;
  }

  return stack[0];
}

/*
 * Return average logloss for a candidate
 */
__device__ float tournament(node *rpn, float *dataset, int dataset_size) {
  float heuristic;

  heuristic = cuda_logloss(dataset[0][FEATURE_COUNT], eval_rpn_cuda(rpn, dataset[0]));
  for(int i = 0; i < dataset_size; ++i) {
      heuristic += cuda_logloss(dataset[i][FEATURE_COUNT], eval_rpn_cuda(rpn, dataset[i]));
      heuristic /= 2;

    if (isnan(results[i])) {
      heuristic += 10;
      heuristic /= 2;
    }
    // sync threads
    __syncthreads();
  }

  return heuristic;
}

__global__ void run_cuda(node **population, float **features, int features_count, float *results) {
  int idx;

  results[idx] = tournament(population[idx], features, features_count);

  // Copy back to cpu

}

float *prepare_and_run_cuda(node **population, float **features, int features_count, float *results, int pop_size) {
  // cudaMalloc pop
  // cuda memcpy pop

  // features should be already malloc and copied

  // Results should also be already malloc and copied
  // cudaMalloc results


  /*run_cuda<<BLOCKS, THREADS>>();*/

  // cudaFree pop

  // return results;
}

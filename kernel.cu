#include "kernel.cuh"

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

__device__ char operation_label_cuda(int operation) {
  if (operation == 0) {
    return '+';
  }
  if (operation == 1) {
    return '*';
  }
  if (operation == 2) {
    return '/';
  }
  if (operation == 3) {
    return '-';
  }
  if (operation == 4) {
    return 's';
  }
  if (operation == 5) {
    return 'f';
  }
  if (operation == 6) {
    return 'l';
  }
  if (operation == 7) {
    return 'e';
  }
  if (operation == 8) {
    return 'c';
  }
  if (operation == 9) {
    return 's';
  }

  return '!';
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

__device__ void display_rpn_cuda(node *rpn) {
  int i = 0;

  while (rpn[i].operation != -2) {
    if (rpn[i].operation != -1) {
      printf(" %c", operation_label_cuda(rpn[i].operation));
    }
    else {
      printf(" %d", rpn[i].feature);
    }
    i++;
  }
  printf("\n");
}

__device__ void display_feature_line_cuda(float *line, int feature_count) {
  int i = 0;

  for (i = 0; i < feature_count; i++) {
    printf(" %f,", line[i]);
  }
  printf("\n");
}


/*
 * Return average logloss for a candidate
 */
__device__ float tournament_cuda(node *rpn, float *dataset, int dataset_size) {

  // Very first element
  /*float heuristic = logloss_cuda(dataset[0 + FEATURE_COUNT], eval_rpn_cuda(rpn, dataset));*/
  /*if (isnan(heuristic))*/
      /*heuristic = 10;*/

  float result = 0.0;
  double heuristic = 0.0;

  for(int i = 0; i < dataset_size; ++i) {
    result = logloss_cuda(dataset[i * FEATURE_COUNT + FEATURE_COUNT],  // Access last element of line
                          eval_rpn_cuda(rpn, dataset + (i * FEATURE_COUNT))); // Set pointer to first element of line

    /*result = eval_rpn_cuda(rpn, dataset + (i * FEATURE_COUNT)); // Set pointer to first element of line*/
    /*if (i < 2) {*/
    /*printf("CUDA Result for %d is %f : %f\n", i, result, heuristic);*/
      /*display_rpn_cuda(rpn);*/
      /*display_feature_line_cuda(dataset + (i * FEATURE_COUNT), 50);*/
      /*printf("\n\n\n");*/
    /*}*/

    /*printf("Result is %f\n", result);*/
    // TODO Clamp
    if (isnan(result) || isinf(result) || result < 0)
      heuristic += 10;
    else
      heuristic += result;

    /*if (i > 0)*/
      /*heuristic /= 2;*/
    // sync threads
    /*__syncthreads();*/
  }

  return heuristic / dataset_size;
}

__global__ void run_cuda(node *population, int pop_size, float *features, int features_count, float *results) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < pop_size)
    results[idx] = tournament_cuda(population + (idx * 256), features, DATASET_SIZE);
}

extern "C"
void prepare_and_run_cuda(node *population, float *d_features, int features_count, float *d_results, int pop_size, float *results_cuda) {
  // cudaMalloc pop
  // cuda memcpy pop

  // Copy population
  node *d_population;
  cudaMalloc(&d_population, (pop_size * 256) * sizeof(node));
  cudaMemcpy(d_population, population, pop_size * 256 * sizeof(node), cudaMemcpyHostToDevice);

  // features should be already malloc and copied

  // Results should also be already malloc and copied
  // flatten
  printf("CUDA RUN\n");
  cudaDeviceSynchronize();
  run_cuda<<<BLOCKS, THREADS>>>(d_population, pop_size, d_features, FEATURE_COUNT, d_results);

  cudaMemcpy(results_cuda, d_results, sizeof(float) * pop_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("CUDA Error: %s\n", cudaGetErrorString(err));

  cudaFree(d_population);
}

/*__host__ void copy_features_cuda(const float **features, int rows, int cols, float *d_features) {*/
  /*int idx = 0;*/
  /*int x = 0;*/

  /*float *features_flatten = (float *)malloc(sizeof(float) * rows * cols);*/

  /*// flatten*/
  /*for (int i = 0; i < rows; i++) {*/
    /*for (x = 0; x < cols; x++) {*/
      /*features_flatten[idx] = features[i][x];*/
      /*idx++;*/
    /*}*/
  /*}*/

  /*cudaMalloc((void**)&d_features, (rows * cols) * sizeof(float));*/
  /*cudaMemcpy(d_features, features_flatten, rows * cols * sizeof(float), cudaMemcpyHostToDevice);*/

  /*free(features_flatten);*/
/*}*/

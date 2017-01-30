#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "node.h"
#include "generator.h"
#include "evaluator.h"
#include "feature_parser.h"
#include <pthread.h>
#include "thread.h"
#include "utils.h"

#include "kernel.cuh"
#include "constants.h"
// TODO RECALC ONLY MODIFIED TREES
// TODO CUDA
//          - Copy results back to device
//          - Check if results match cpu results
//
//
//
//
// TODO JS INTERPRETER

static timestamp_t get_timestamp() {
  struct timeval now;
  gettimeofday (&now, NULL);

  return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

void copy_features_cuda(const float **features, int rows, int cols, float *d_features) {
  int idx = 0;
  int x = 0;

  float *features_flatten = (float *)malloc(sizeof(float) * rows * cols);

  // flatten
  for (int i = 0; i < rows; i++) {
    for (x = 0; x < cols; x++) {
      features_flatten[idx] = features[i][x]; idx++;
    }
  }

  cudaMalloc((void**)&d_features, (rows * cols) * sizeof(float));
  cudaMemcpy(d_features, features_flatten, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

  free(features_flatten);
}

node *pop_to_1d(node **rpn_population, int population_count) {
  int idx = 0;
  int x = 0;

  node *rpn_population_1d = (node *)malloc(sizeof(node) * population_count * TREE_SIZE);

  for (int i = 0; i < population_count; i++) {
    for (x = 0; x < TREE_SIZE; x++) {
      rpn_population_1d[idx] = rpn_population[i][x];
      idx++;
    }
  }

  return rpn_population_1d;
}

void pop_to_rpn(node **population, int population_count, node** rpn_population) {
  int pos;
  for(int i = 0; i < population_count; i++) {
    rpn_population[i] = (node *)malloc(sizeof(node) * TREE_SIZE);
    pos = 0;
    tree_to_rpn(population[i], 0, rpn_population[i], &pos);
    rpn_population[i][pos].operation = -2;
  }
}

int rmdup(int *array, int length) {
  int *current , *end = array + length - 1;
  int removed = 0;

  for (current = array + 1; array < end; array++, current = array + 1) {
    while (current <= end) {
      if (*current == *array) {
        *current = *end--;
        removed++;
      }
      else {
        current++;
      }
    }
  }

  return removed;
}

int losers_size(int *losers) {
  int i = 0;
  while (losers[i] != -1) {
    i++;
  }

  return i;
}

void crossover(node **population, int population_count, float *results, int *all_losers, float percentage) {
  int max_pop = floor(population_count * (percentage / 2));
  int i = 0;
  int losers_idx = 0;

  int winners[2];
  int losers[2];
  int *compete = (int *)malloc(sizeof(int) * 4);

  int offsetFrom = 0;
  int offsetTo = 0;

  /*printf("Max pop is %d\n", max_pop);*/
  for(i = 0; i < max_pop; i++) {
    uniq_rand(compete, population_count, 0);

    winners[0] = minIndex(results, compete[0], compete[1]);
    losers[0]  = maxIndex(results, compete[0], compete[1]);
    all_losers[losers_idx] = losers[0];
    losers_idx++;

    winners[1] = minIndex(results, compete[2], compete[3]);
    losers[1]  = maxIndex(results, compete[2], compete[3]);
    all_losers[losers_idx] = losers[0];
    losers_idx++;

    /*puts("After min");*/

    /*puts("Before min");*/
    /*printf("Memcy %d, %d\n", losers[0], winners[0]);*/
    /*memcpy(population[losers[0]], population[winners[0]], sizeof(node) * TREE_SIZE);*/
    copy_branch(population, winners[0], losers[0], 0, 0);

    offsetFrom = random_subtree(population[winners[1]], 0, 2, 20);
    offsetTo = random_subtree(population[losers[0]], 0, 2, 20);
    copy_branch(population, winners[1], losers[0], offsetFrom, offsetTo);



    // Copy parent
    copy_branch(population, winners[1], losers[1], 0, 0);

    offsetFrom = random_subtree(population[winners[0]], 0, 1, 60);
    offsetTo = random_subtree(population[losers[1]], 0, 1, 60);
    copy_branch(population, winners[0], losers[1], offsetFrom, offsetTo);
  }

  // Remove duplicates
  int removed = rmdup(all_losers, losers_idx);
  all_losers[losers_idx - removed] = -1;
  printf("Losers count %d %d\n", losers_size(all_losers), losers_idx);
  free(compete);
}


void mutate(node **population, int population_count, float percentage) {
  int max_pop = floor(population_count * percentage);
  int i = 0;
  int node_index = 0;

  for(i = 0; i < max_pop; i++) {
    node_index = rand() % population_count;

    mutate_tree(population, node_index, 0, 50);
  }
}

void clone(node **population, node *model, int toIndex) {
  memcpy(population[toIndex], model, sizeof(node) * TREE_SIZE);
}

double logloss(double actual, double predicted) {
  double eps = 1e-15;

  if (predicted < eps)
    predicted = eps;

  if (predicted == 1)
    predicted -= eps;

  return - (actual * logf(predicted) + (1 - actual) * logf(1 - predicted));
}

void tournament(node **population, int population_count, const float **dataset, int start, int end, float *results) {
  float heuristic = 0;
  float result = 0;
  int i = 0;
  int x = 0;

  for(i = 0; i < population_count; i++) {
    for(x = start; x < end; x++) {
      result = logloss(dataset[x][FEATURE_COUNT], eval_rpn(population[i], dataset[x]));
      /*result = eval_rpn(population[i], dataset[x]);*/

      if (isnan(result) || isinf(result) || result < 0)
        heuristic += 10;
      else
        heuristic += result;
    }

    results[i] = heuristic / ((end - start));
    heuristic = 0;
  }

  /*puts("Return tournament");*/
  return ;
}

void *thread_wrapper(void *data) {
  gpthread* gp = (gpthread *) data;

  tournament(gp->population, gp->population_count, gp->features, gp->start, gp->end, gp->results);

  return NULL;
}

float *concat(gpthread **gp, float * results) {
  float result = 0;

  for (int i = 0; i < POPULATION_SIZE; i++) {
    result = 0;
    for (int j = 0; j < NUMTHREADS; j++) {
      result += gp[j]->results[i];
    }

    results[i] = result / NUMTHREADS;
  }

  return results;
}

void pick_pop(node **population, int *indexes, node **picked) {
  int i = 0;

  while(indexes[i] != -1) {
    memcpy(picked[i], population[indexes[i]], TREE_SIZE * sizeof(node));
    i++;
  }
}

void run() {
  if (THREADS * BLOCKS != POPULATION_SIZE) {
    printf("ERROR POPULATION_SIZE AND BLOCKS ARE DIFFERENT SIZE\n");
    exit(-1);
  }

  gpthread **gp = (gpthread **)malloc(sizeof(gpthread *) * NUMTHREADS);

  FILE *datasetFile = fopen("./data/numerai_training_data.csv", "r");
  float const **featuresPtr = (float const **)feature_fromFile(datasetFile, DATASET_SIZE, FEATURE_COUNT);
  fclose(datasetFile);

  puts("Copy features to device");

  //float *d_features;
  //copy_features_cuda(featuresPtr, DATASET_SIZE, FEATURE_COUNT, d_features);

  float *d_results;
  float *results_cuda = (float *)malloc(sizeof(float) * POPULATION_SIZE);
  cudaMalloc((void **)&d_results, POPULATION_SIZE * sizeof(float));

  int i = 0;
  int max_depth = MAX_DEPTH;
  int generations = 300; // 10k
  int population_count = POPULATION_SIZE;
  int display_count = 0;

  puts("Malloc pop for 10k generation (500 pop)");

  node **population = (node **)malloc(sizeof(node *) * population_count);
  node **population_losers = (node **)malloc(sizeof(node *) * population_count);
  for(i = 0; i < population_count; i++) {
    population[i] = (node *)malloc(sizeof(node) * TREE_SIZE);
  }

  puts("Generating pop for 10k generation (500 pop)");
  timestamp_t t1 = get_timestamp();
  for(i = 0; i < population_count; i++) {
    population[i][0].operation = rand() % OPERATION_COUNT;
    generate_tree(population[i], 1, 1, max_depth, FEATURE_COUNT);
    generate_tree(population[i], 2, 1, max_depth, FEATURE_COUNT);
  }

  // Copy population to rpn representation
  node **rpn_population = (node **)malloc(sizeof(node *) * population_count);

  pop_to_rpn(population, population_count, rpn_population);

  node *rpn_1d;
  /*node *rpn_1d = pop_to_1d(rpn_population, POPULATION_SIZE);*/

  /*t1 = get_timestamp();*/
  /*prepare_and_run_cuda(rpn_1d, featuresPtr, FEATURE_COUNT, d_results, POPULATION_SIZE, results_cuda);*/
  timestamp_t t2 = get_timestamp();
  double secs = (t2 - t1) / 1000000.0L;


  /*printf("CUDA RPN TOOK: %.5f s\n", secs);*/

  int idx = 0;
  int x = 0;
  float *d_features;
  float *features_flatten = (float *)malloc(sizeof(float) * DATASET_SIZE * FEATURE_COUNT);

  // flatten
  for (int i = 0; i < DATASET_SIZE; i++) {
	  for (x = 0; x < FEATURE_COUNT; x++) {
		  features_flatten[idx] = featuresPtr[i][x];
		  idx++;
	  }
  }


  cudaMalloc((void**)&d_features, (DATASET_SIZE * FEATURE_COUNT) * sizeof(float));
  cudaMemcpy(d_features, features_flatten, DATASET_SIZE * FEATURE_COUNT * sizeof(float), cudaMemcpyHostToDevice);


  /*quicksort(results_cuda, population_count);*/
  /*display_top(results_cuda, 10);*/

  for (i = 0; i < NUMTHREADS; i++) {

    gp[i] = (gpthread *)malloc(sizeof(gpthread));
    gp[i]->features = featuresPtr;
    gp[i]->population_count = POPULATION_SIZE;
    gp[i]->results = (float *)malloc(POPULATION_SIZE * sizeof(float));
    /*gp[i]->population = population;*/
    gp[i]->population = rpn_population;
    gp[i]->start = i * (DATASET_SIZE / NUMTHREADS);
    gp[i]->end   = gp[i]->start + (DATASET_SIZE / NUMTHREADS);

    if (gp[i]->end > DATASET_SIZE) {
      gp[i]->end = DATASET_SIZE;
    }

    printf("Thread %d prepared:start from %d, end at %d\n", i, gp[i]->start, gp[i]->end);
  }

  float *results = (float *)malloc(population_count * sizeof(float));
  float *results_cpy = (float *)malloc(population_count * sizeof(float));
  int *losers = (int *)malloc(population_count * sizeof(int));
  int losers_count = 0;
  int pop = 0;
  puts("Generating pop done for 5M individuals");

  timestamp_t t3, t4, t5, t6;
  pthread_t thread[NUMTHREADS];

  /*node *line = malloc(sizeof(node) * 256);*/

  for(int y = 0; y < generations; y++) {


    // launch kernel
    t3 = get_timestamp();
    if (y == 0) {
      pop = POPULATION_SIZE; // should be replace with losers size
    }
    else {
      pop = losers_count;
    }
    rpn_1d = pop_to_1d(rpn_population, pop);
    prepare_and_run_cuda(rpn_1d, d_features, FEATURE_COUNT, d_results, pop, results_cuda);

    t4 = get_timestamp();
    // MERGE results_cpy back
    //
    memcpy(results_cpy, results_cuda, pop * sizeof(float));
    quicksort(results_cuda, pop);


    // Start threads
    /*for (i = 0; i < NUMTHREADS; i++) {*/
      /*pthread_create(&thread[i], NULL, thread_wrapper, gp[i]);*/
    /*}*/

    /*// Wait threads to finish*/
    /*for (i = 0; i < NUMTHREADS; i++) {*/
      /*pthread_join(thread[i], NULL);*/
    /*}*/

    /*t4 = get_timestamp();*/

    /*t5 = get_timestamp();*/
    /*// Concat*/
    /*concat(gp, results);*/

    /*memcpy(results_cpy, results, population_count * sizeof(float));*/

    /*quicksort(results, population_count);*/
    /*float threshold = percentile(results, population_count, 0.9); // 90%*/
    /*printf("Low Quartile is: %f\n", threshold);*/
    /*printf("Best: %lf\n", results_cpy[0]);*/

    double t_secs = (t4 - t3) / 1000000.0L;

    /*selection(population, population_count, featuresPtr, DATASET_SIZE, results_cpy, threshold);*/
    t5 = get_timestamp();
    crossover(population, POPULATION_SIZE, results_cpy, losers, 0.9);
    losers_count = losers_size(losers);
    pick_pop(population, losers, population_losers);
    

    // Keep index of cross
    // recalc these
    // merge
    mutate(population, population_count, 0.1); // keep track of mutated
    pop_to_rpn(population, population_count, rpn_population);
    pop_to_rpn(population_losers, losers_count, rpn_population);
    // to 1d
    /*rpn_1d;*/
    // launch kernel

    t6 = get_timestamp();
    /*if (display_count > 1) {*/
    /*printf("Score: %f\n", naive_average(results_cuda, population_count));*/
    display_top(results_cuda, 10);
    /*pos = 0;*/
    /*tree_to_rpn(population[0], 0, line, &pos);*/
    /*line[pos].operation = -2;*/
    /*display_rpn(line, pos);*/
    /*t5 = get_timestamp();*/
    /*for (i = 0; i < 10000; i++) {*/
      /*eval_rpn(line, featuresPtr[0]);*/
    /*};*/
    /*printf("RPN RESULT IS %f\n",   logloss(featuresPtr[0][FEATURE_COUNT], eval_rpn(line, featuresPtr[0])));*/
    /*t6 = get_timestamp();*/
    double t_secs2 = (t6 - t5) / 1000000.0L;
    /*printf("RPN took: %.6f ms\n", t_secs2);*/
    /*fflush(stdout);*/
    /*t5 = get_timestamp();*/
    /*for (i = 0; i < 10000; i++) {*/
      /*evaluate_tree(population[0], 0, featuresPtr[0]);*/
    /*};*/
    /*printf("TREE RESULT IS %f\n",  logloss(featuresPtr[0][FEATURE_COUNT], evaluate_tree(population[0], 0, featuresPtr[0])));*/
    /*t6 = get_timestamp();*/
    /*t_secs2 = (t6 - t5);*/
    /*printf("TREE took: %.6f ms\n", t_secs2);*/
    /*display_count = 0;*/
    printf("Generation : %d\n", y);
    printf("Tournament took: %.5f s\n", t_secs);
    printf("Crossover/Mutation took: %.5f s\n", t_secs2);
    /*}*/

    /*display_count++;*/
  }

  /*for (i = 0; i < NUMTHREADS; i++) {*/
    /*free(gp[i]->results);*/
  /*}*/

  /*free(results);*/
  /*free(results_cpy);*/

  t2 = get_timestamp();
  secs = (t2 - t1) / 1000000.0L;
  printf("Backtest took: %.3f s\n", secs);

  return ;

}

int main(int argc, char **argv) {
  srand(time(NULL));

  /*FILE *logFile = fopen("./debug.json", "w");*/

  run();
  /*fclose(logFile);*/
  return 0;
}

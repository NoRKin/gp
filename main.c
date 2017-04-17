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
#include "population.h"

#include "kernel.cuh"
#include "constants.h"
// TODO RECALC ONLY MODIFIED TREES [x]
// TODO MULTICORE - Mutation
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

int losers_size(int *losers) {
  int i = 0;
  while (losers[i] != -1) {
    i++;
  }

  return i;
}

void crossover(node **population, int population_count, float *results, int *all_losers, float percentage) {
  int max_pop = floor(population_count * (percentage)) / 2;
  int i = 0;
  int losers_idx = 0;

  int winners[2];
  int losers[2];
  int *compete = (int *)malloc(sizeof(int) * 4);

  int offsetFrom = 0;
  int offsetTo = 0;

  printf("Max pop is %d\n", max_pop);
  for(i = 0; i < max_pop; i += 2) {
    uniq_rand(compete, population_count, 0);

    winners[0] = minIndex(results, compete[0], compete[1]);
    losers[0]  = maxIndex(results, compete[0], compete[1]);
    all_losers[losers_idx] = losers[0];
    losers_idx++;

    winners[1] = minIndex(results, compete[2], compete[3]);
    losers[1]  = maxIndex(results, compete[2], compete[3]);
    all_losers[losers_idx] = losers[1];
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
  //int removed = rmdup(all_losers, losers_idx); // LOSING ALL THE FREAKING INDEXES
  //all_losers[losers_idx - removed] = -1;
  all_losers[losers_idx] = -1;
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

  /*printf("Evaluating %d\n", population_count);*/

  for(i = 0; i < population_count; i++) {
    for(x = start; x < end; x++) {
      result = logloss(dataset[x][FEATURE_COUNT], eval_rpn(population[i], dataset[x]));
      /*result = eval_rpn(population[i], dataset[x]);*/
      /*result = logloss(dataset[x][FEATURE_COUNT], evaluate_tree(population[i], 0, dataset[x]));*/

      if (isnan(result) || isinf(result) || result < 0)
        heuristic += 10;
      else
        heuristic += result;
    }

    // Average all rows
    results[i] = heuristic / ((end - start));
    heuristic = 0;
  }

  /*puts("Return tournament");*/
  return ;
}

void *thread_wrapper(void *data) {
  gpthread* gp = (gpthread *) data;

  // Build pop from losers
  tournament(gp->population, gp->population_count, gp->features, gp->start, gp->end, gp->results);

  return NULL;
}

void pick_pop(node **population, int *indexes, node **picked) {
  int i = 0;
  int j = 0;
  int offset;

  while(indexes[i] != -1) {
    /*printf("Pick index %d - %d : op %d\n", i, indexes[i], population[indexes[i]][0].operation);*/
    /*copy_tree(population[indexes[i]], picked[i], 0);*/
    offset = 0;
    tree_to_rpn(population[indexes[i]], 0, picked[i], &offset);
    picked[i][offset].operation = -2;
    /*for (j = 0; j < 256; j++) {*/
    /*picked[i][j].operation = population[indexes[i]][j].operation;*/
    /*picked[i][j].feature = population[indexes[i]][j].feature;*/
    /*}*/
    /*memcpy(picked[i], population[indexes[i]], TREE_SIZE * sizeof(node));*/
    i++;
  }
  puts("COPY DONE");
}

void merge_back_results(float *results, float *new_results, int *indexes) {
  int i = 0;

  while(indexes[i] != -1) {
    results[indexes[i]] = new_results[i];
    i++;
  }

  printf("%d results merged back\n", i);
}

void tournament_cpu(gpthread **gp, float *results) {
  pthread_t thread[NUMTHREADS];

  for (int i = 0; i < NUMTHREADS; i++) {
    pthread_create(&thread[i], NULL, thread_wrapper, gp[i]);
  }

  /*// Wait threads to finish*/
  for (int i = 0; i < NUMTHREADS; i++) {
    pthread_join(thread[i], NULL);
  }
}


void run_gp_cpu(node **population, node **population_losers, gpthread **gp) {
  timestamp_t t1, t2, t3, t4;
  pthread_t thread[NUMTHREADS];

  // Each thread struct points towards the same population
  node **rpn_population = (node **)malloc(sizeof(node *) * POPULATION_SIZE);
  pop_to_rpn(population, POPULATION_SIZE, rpn_population, TREE_SIZE);
  for (int i = 0; i < NUMTHREADS; i++) {
    gp[i]->population = rpn_population;
    /*gp[i]->population = population;*/
  }

  float *results = (float *)malloc(POPULATION_SIZE * sizeof(float));
  int *losers = (int *)malloc(POPULATION_SIZE * sizeof(int));
  int losers_count = POPULATION_SIZE;

  tournament_cpu(gp, results);
  concat(gp, results, POPULATION_SIZE, NUMTHREADS);

  for(int y = 0; y < GENERATIONS; y++) {
    display_top(results, POPULATION_SIZE, 10);
    // Reduce our population to only winners
    // Losers index becomes useless
    t3 = get_timestamp();
    crossover(population, POPULATION_SIZE, results, losers, CROSSOVER_RATE);
    mutate(population, POPULATION_SIZE, MUTATION_RATE); // keep track of mutated

    // Take loosers from population
    losers_count = losers_size(losers);
    pick_pop(population, losers, population_losers);
    /*pop_to_rpn_alt(population_losers, losers_count, rpn_population, TREE_SIZE);*/

    // Assign new population
    for (int i = 0; i < NUMTHREADS; i++) {
      gp[i]->population_count = losers_count;
      gp[i]->population = population_losers;
      /*gp[i]->population = rpn_population;*/
    }
    t4 = get_timestamp();

    t1 = get_timestamp();
    tournament_cpu(gp, results);
    t2 = get_timestamp();

    // Merge results back
    float *tmp_results = malloc(sizeof(float) * losers_count);
    concat(gp, tmp_results, losers_count, NUMTHREADS);
    merge_back_results(results, tmp_results, losers);
    free(tmp_results);

    double t_secs = (t2 - t1) / 1000000.0L;
    double t_secs2 = (t4 - t3) / 1000000.0L;
    printf("Generation : %d\n", y);
    printf("Tournament took: %.5f s\n", t_secs);
    printf("Crossover/Mutation took: %.5f s\n", t_secs2);
  }
}


void run_gp_gpu(node **population, node **population_losers, float *d_features, float *d_results, node *rpn_1d) {
  timestamp_t t1, t2;

  float *results = (float *)malloc(sizeof(float) * POPULATION_SIZE);
  float *results_cuda = (float *)malloc(sizeof(float) * POPULATION_SIZE);

  int *losers = (int *)malloc(POPULATION_SIZE * sizeof(int));
  int losers_count = POPULATION_SIZE;

  prepare_and_run_cuda(rpn_1d, d_features, FEATURE_COUNT, d_results, POPULATION_SIZE, results_cuda);
  memcpy(results, results_cuda, POPULATION_SIZE * sizeof(float));

  for(int y = 0; y < GENERATIONS; y++) {
    display_top(results, POPULATION_SIZE, 10);
    crossover(population, POPULATION_SIZE, results, losers, CROSSOVER_RATE);
    mutate(population, POPULATION_SIZE, MUTATION_RATE); // keep track of mutated

    // Get losers
    losers_count = losers_size(losers);
    pick_pop(population, losers, population_losers);

    // Rebuild rpn
    rpn_1d = pop_to_1d(population_losers, losers_count, TREE_SIZE);

    t1 = get_timestamp();
    prepare_and_run_cuda(rpn_1d, d_features, FEATURE_COUNT, d_results, losers_count, results_cuda);
    t2 = get_timestamp();

    // Merge back to results
    merge_back_results(results, results_cuda, losers);
    printf("Generation : %d\n", y + 1);
    printf("Tournament took: %.5f s\n", (double)((t2 - t1) / 1000000.0L));
  }
}

void prepare_gpu(const float **featuresPtr, node **population, node **population_losers) {
  // Create first population
  init_population(population, POPULATION_SIZE, OPERATION_COUNT, FEATURE_COUNT, MAX_DEPTH);

  // Prepare cuda populations
  float *d_results;
  cudaMalloc((void **)&d_results, POPULATION_SIZE * sizeof(float));

  // Copy population to rpn representation
  node **rpn_population = (node **)malloc(sizeof(node *) * POPULATION_SIZE);
  pop_to_rpn(population, POPULATION_SIZE, rpn_population, TREE_SIZE);


  // Flatten features
  float *features_1d = flatten_features(featuresPtr, DATASET_SIZE, FEATURE_COUNT);

  // Allocate features on device
  float *d_features;
  cudaMalloc((void**)&d_features, (DATASET_SIZE * FEATURE_COUNT) * sizeof(float));
  cudaMemcpy(d_features, features_1d, DATASET_SIZE * FEATURE_COUNT * sizeof(float), cudaMemcpyHostToDevice);

  node *rpn_1d = pop_to_1d(rpn_population, POPULATION_SIZE, TREE_SIZE);

  // tournament
  run_gp_gpu(population, population_losers, d_features, d_results, rpn_1d);
}

gpthread **prepare_cpu(const float **featuresPtr, node **population, node **population_losers) {
  gpthread **gp = (gpthread **)malloc(sizeof(gpthread *) * NUMTHREADS);

  printf("Malloc pop for %d generation (%d pop)\n", GENERATIONS, POPULATION_SIZE);

  // Prepare memory allocation for each thread
  for (int i = 0; i < NUMTHREADS; i++) {
    gp[i] = (gpthread *)malloc(sizeof(gpthread));
    gp[i]->features = featuresPtr;
    gp[i]->population_count = POPULATION_SIZE;
    gp[i]->results = (float *)malloc(POPULATION_SIZE * sizeof(float));
    /*gp[i]->population = rpn_population;*/
    gp[i]->start = i * (DATASET_SIZE / NUMTHREADS);
    gp[i]->end = gp[i]->start + (DATASET_SIZE / NUMTHREADS);

    // Stay within memory bound
    if (gp[i]->end > DATASET_SIZE) {
      gp[i]->end = DATASET_SIZE;
    }

    printf("Thread %d prepared:start from %d, end at %d\n", i, gp[i]->start, gp[i]->end);
  }

  return gp;
}

int main(int argc, char **argv) {
  srand(time(NULL));

  if (THREADS * BLOCKS != POPULATION_SIZE) {
    printf("ERROR POPULATION_SIZE AND BLOCKS ARE DIFFERENT SIZE\n");
    exit(-1);
  }

  FILE *datasetFile = fopen("./data/numerai_training_data.csv", "r");
  float const **featuresPtr = (float const **)feature_fromFile(datasetFile, DATASET_SIZE, FEATURE_COUNT);
  fclose(datasetFile);

  // Alloc Population
  node **population = create_population(POPULATION_SIZE, TREE_SIZE);
  node **population_losers = create_population(POPULATION_SIZE, TREE_SIZE);

  // Create first population
  init_population(population, POPULATION_SIZE, OPERATION_COUNT, FEATURE_COUNT, MAX_DEPTH);

  if (USE_CUDA) {
    prepare_gpu(featuresPtr, population, population_losers);
  }
  else {
    // Encapsulate all required data in gp structure
    gpthread **gp = prepare_cpu(featuresPtr, population, population_losers);
    run_gp_cpu(population, population_losers, gp);
  }

  return 0;
}

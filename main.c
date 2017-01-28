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

// TODO RECALC ONLY MODIFIED TREES
// TODO CUDA
//        -- Cudify operations
//        -- cudaMalloc features
//        -- cudaMalloc pop
//
// TODO JS INTERPRETER

#define DEBUG 0
#define MAX_DEPTH 4
#define DATASET_SIZE 100000
#define FEATURE_COUNT 50
#define TREE_SIZE 256
#define POPULATION_SIZE 1000
#define NUMTHREADS 8

static timestamp_t get_timestamp() {
  struct timeval now;
  gettimeofday (&now, NULL);

  return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

void pop_to_rpn(node **population, int population_count, node** rpn_population) {
  int pos;
  for(int i = 0; i < population_count; i++) {
    rpn_population[i] = malloc(sizeof(node) * TREE_SIZE);
    pos = 0;
    tree_to_rpn(population[i], 0, rpn_population[i], &pos);
    rpn_population[i][pos].operation = -2;
  }
}

void crossover(node **population, int population_count, float *results, float percentage) {
  int max_pop = floor(population_count * percentage);
  int i = 0;
  int x = 0;

  int winners[2];
  int losers[2];
  int *compete = malloc(sizeof(int) * 4);

  int offset = 0;
  int depth = 0;

  int offsetFrom = 0;
  int offsetTo = 0;

  /*printf("Max pop is %d\n", max_pop);*/
  for(i = 0; i < max_pop; i++) {
    uniq_rand(compete, population_count, 0);

    winners[0] = minIndex(results, compete[0], compete[1]);
    losers[0]  = maxIndex(results, compete[0], compete[1]);

    winners[1] = minIndex(results, compete[2], compete[3]);
    losers[1]  = maxIndex(results, compete[2], compete[3]);
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

  free(compete);
}


void mutate(node **population, int population_count, float *results, float percentage) {
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

  return - (actual * log(predicted) + (1 - actual) * log(1 - predicted));
}

void tournament(node **population, int population_count, const float **dataset, int dataset_size, int start, int end, float *results) {
  long double heuristic = 0;
  int i = 0;
  int x = 0;

  for(i = 0; i < population_count; i++) {
    for(x = start; x < end; x++) {
      /*heuristic += logloss(dataset[x][FEATURE_COUNT], evaluate_tree(population[i], 0, dataset[x]));*/
      heuristic += logloss(dataset[x][FEATURE_COUNT], eval_rpn(population[i], dataset[x]));
      /*printf("Heuristic is %f\n", evaluate_tree(population[i], 0, dataset[x]));*/
    }
    results[i] = heuristic / (dataset_size / NUMTHREADS);
    if (isnan(results[i])) {
      results[i] = 10;
    }
    heuristic = 0;
  }

  return ;
}

void *thread_wrapper(void *data) {
  gpthread* gp = (gpthread *) data;

  tournament(gp->population, gp->population_count, gp->features, DATASET_SIZE, gp->start, gp->end, gp->results);
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


void run(FILE *logFile) {
  gpthread **gp = malloc(sizeof(gpthread *) * 4);

  FILE *datasetFile = fopen("./data/numerai_training_data.csv", "r");
  float const **featuresPtr = (float const **)feature_fromFile(datasetFile, DATASET_SIZE, FEATURE_COUNT);

  int i = 0;
  int max_depth = MAX_DEPTH;
  int generations = 300; // 10k
  int population_count = POPULATION_SIZE;
  int display_count = 0;

  puts("Malloc pop for 10k generation (500 pop)");

  node **population = malloc(sizeof(node *) * population_count);
  for(i = 0; i < population_count; i++) {
    population[i] = malloc(sizeof(node) * TREE_SIZE);
  }

  puts("Generating pop for 10k generation (500 pop)");
  timestamp_t t1 = get_timestamp();
  for(i = 0; i < population_count; i++) {
    population[i][0].operation = rand() % OPERATION_COUNT;
    generate_tree(population[i], 1, 1, max_depth, FEATURE_COUNT);
    generate_tree(population[i], 2, 1, max_depth, FEATURE_COUNT);
  }

  // Copy population to rpn representation
  node **rpn_population = malloc(sizeof(node *) * population_count);

  t1 = get_timestamp();
  pop_to_rpn(population, population_count, rpn_population);
  timestamp_t t2 = get_timestamp();
  double secs = (t2 - t1) / 1000000.0L;
  printf("Pop to rpn took: %.5f s\n", secs);


  for (i = 0; i < NUMTHREADS; i++) {

    gp[i] = malloc(sizeof(gpthread));
    gp[i]->features = featuresPtr;
    gp[i]->population_count = POPULATION_SIZE;
    gp[i]->results = malloc(POPULATION_SIZE * sizeof(float));
    /*gp[i]->population = population;*/
    gp[i]->population = rpn_population;
    gp[i]->start = i * (DATASET_SIZE / NUMTHREADS);
    gp[i]->end   = gp[i]->start + (DATASET_SIZE / NUMTHREADS);

    if (gp[i]->end > DATASET_SIZE) {
      gp[i]->end = DATASET_SIZE;
    }

    printf("Thread %d prepared:start from %d, end at %d\n", i, gp[i]->start, gp[i]->end);
  }

  float *results = malloc(population_count * sizeof(float));
  float *results_cpy = malloc(population_count * sizeof(float));
  puts("Generating pop done for 5M individuals");

  timestamp_t t3, t4, t5, t6;
  pthread_t thread[NUMTHREADS];

  /*node *line = malloc(sizeof(node) * 256);*/

  for(int y = 0; y < generations; y++) {
    t3 = get_timestamp();

    // Start threads
    for (i = 0; i < NUMTHREADS; i++) {
      pthread_create(&thread[i], NULL, thread_wrapper, gp[i]);
    }

    // Wait threads to finish
    for (i = 0; i < NUMTHREADS; i++) {
      pthread_join(thread[i], NULL);
    }
    t4 = get_timestamp();

    t5 = get_timestamp();
    // Concat
    concat(gp, results);

    memcpy(results_cpy, results, population_count * sizeof(float));

    quicksort(results, population_count);
    /*float threshold = percentile(results, population_count, 0.9); // 90%*/
    /*printf("Low Quartile is: %f\n", threshold);*/
    /*printf("Best: %lf\n", results_cpy[0]);*/

    double t_secs = (t4 - t3) / 1000000.0L;

    /*selection(population, population_count, featuresPtr, DATASET_SIZE, results_cpy, threshold);*/
    crossover(population, population_count, results_cpy, 0.4);
    mutate(population, population_count, results_cpy, 0.1);
    pop_to_rpn(population, population_count, rpn_population);

    t6 = get_timestamp();
    /*if (display_count > 1) {*/
    printf("Score: %f\n", naive_average(results, population_count));
    display_top(results, 10);
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
    double t_secs2 = (t6 - t5);
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

  free(results);
  free(results_cpy);

  t2 = get_timestamp();
  secs = (t2 - t1) / 1000000.0L;
  printf("Backtest took: %.3f s\n", secs);

  fclose(datasetFile);
}

int main(int argc, char **argv) {
  srand(time(NULL));

  FILE *logFile = fopen("./debug.json", "w");

  run(logFile);
  fclose(logFile);
  return 0;
}

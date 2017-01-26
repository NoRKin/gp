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

typedef unsigned long long timestamp_t;


// TODO JS INTERPRETER
// TODO PARRALLELIZATION

#define MAX_DEPTH 4
#define DATASET_SIZE 96000
#define FEATURE_COUNT 21

static timestamp_t get_timestamp () {
  struct timeval now;
  gettimeofday (&now, NULL);

  return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

void quicksort(float *A, int len)
{
  if (len < 2) return;

  float pivot = A[len / 2];

  int i, j;
  for (i = 0, j = len - 1; ; i++, j--)
  {
    while (A[i] < pivot) i++;
    while (A[j] > pivot) j--;

    if (i >= j) break;

    float temp = A[i];
    A[i]     = A[j];
    A[j]     = temp;
  }

  quicksort(A, i);
  quicksort(A + i, len - i);
}

float naive_average(float *a, int n) {
  float sum = 0;

  for( int i = 0; i < n; i++ )
    sum += a[i];

  return sum / n;
}

float max(float a, float b) {
  if (a > b)
    return a;
  else
    return b;
}

float min(float a, float b) {
  if (a < b)
    return a;
  else
    return b;
}

/*void crossover(node *father, node *mother, node **population, int childIndex) {*/
  // population[childIndex] will become a mix between father, mother
  // traverse father
  // insert random mother node
/*}*/

void crossover(node **population, int population_count, float *results, float percentage) {
  int max = floor(population_count * percentage);
  int i = 0;

  // Choose two individual -> Select the best
  for(i = 0; i < max; i++) {
      Node *father = population[rand() % population_count];

  }
}


void mutate() {
  // Traverse tree

}

void clone(node **population, node *model, int toIndex) {
  memcpy(population[toIndex], model, sizeof(node) * 64);
}

double logloss(double actual, double predicted) {
  /*printf("Actual is %lf, predicted is %lf\n", actual, predicted);*/
  double eps = 1e-15;

  predicted = fabs(atan(predicted));
  if (predicted > 1)
    predicted = 1;

  if (predicted < eps)
    predicted = eps;

  if (predicted == 1)
    predicted -= eps;

  /*printf("Actual is %lf, predicted is %lf\n", actual, predicted);*/
  return - (actual * log(predicted) + (1 - actual) * log(1 - predicted));
}

void selection(node **population, int population_count, const float **dataset, int dataset_size, float *results, float threshold) {
  // Get the 50% best
  float avg = 0.0;
  int *elite = malloc(sizeof(int) * population_count);
  int elite_count = 0;

  // Map elite
  /*avg = naive_average(results, population_count);*/
  for(int i = 0; i < population_count; i++) {
    if (results[i] < threshold) {
      elite[elite_count] = i;
      elite_count++;
    }
  }

  for(int i = 0; i < population_count; i++) {
    if (results[i] > threshold) {
      int reproduce = 0;

      if (reproduce == 1) {
        // TODO: Wrong mother and father are not within the elite
        node *mother = population[elite[rand() % elite_count]];
        node *father = population[elite[rand() % elite_count]];
        /*crossover(father, mother, population, i);*/
      }
      else {
        // Can be a children
        // Or Replace in 20% of the case
        population[i][0].operation = rand() % OPERATION_COUNT;
        generate_tree(population[i], 1, 1, MAX_DEPTH, FEATURE_COUNT);
        generate_tree(population[i], 2, 1, MAX_DEPTH, FEATURE_COUNT);
      }
    }
  }
}


float *tournament(node **population, int population_count, const float **dataset, int dataset_size) {
  float *results = malloc(sizeof(float) * population_count);

  long double heuristic = 0;

  for(int i = 0; i < population_count; i++) {
    // TODO: Parralellize
    for(int x = 0; x < dataset_size; x++) {
      heuristic += logloss(dataset[x][FEATURE_COUNT], evaluate_tree(population[i], 0, dataset[x]));
      /*printf("Heuristic is %f\n", evaluate_tree(population[i], 0, dataset[x]));*/
    }
    results[i] = heuristic / dataset_size;
    if (isnan(results[i])) {
      results[i] = 10;
    }
    /*printf("Heuristic avg is %f\n", results[i]);*/
    heuristic = 0;
  }

  return results;
}

void display_top(float *results, int n) {
  for (int i = 0; i < n; i++) {
    printf("Position: %d\t Score: %f\n", i, results[i]);
  }
}

float percentile(float *results, int length, float top) {
  int portionIndex = floor(length * top);

  return results[portionIndex];
}

void test(FILE *logFile) {
  FILE *datasetFile = fopen("./datasets.csv", "r");
  float const **featuresPtr = (float const **)feature_fromFile(datasetFile, DATASET_SIZE, FEATURE_COUNT);

  int i = 0;
  int max_depth = 3;
  int generations = 1000; // 10k
  int population_count = 100;

  puts("Malloc pop for 10k generation (500 pop)");

  node **population = malloc(sizeof(node *) * population_count);
  for(i = 0; i < population_count; i++) {
    population[i] = malloc(sizeof(node) * 128);
  }

  puts("Generating pop for 10k generation (500 pop)");
  timestamp_t t1 = get_timestamp();
  for(i = 0; i < population_count; i++) {
    population[i][0].operation = rand() % OPERATION_COUNT;
    generate_tree(population[i], 1, 1, max_depth, FEATURE_COUNT);;
    generate_tree(population[i], 2, 1, max_depth, FEATURE_COUNT);
  }

  puts("Generating pop done for 5M individuals");
  float threshold;

  timestamp_t t3, t4;
  for(int y = 0; y < generations; y++) {
    t3 = get_timestamp();
    // TODO: Parrallelize
    float *results = tournament(population, population_count, featuresPtr, DATASET_SIZE);

    float *results_cpy = malloc(population_count * sizeof(float));
    memcpy(results_cpy, results, population_count * sizeof(float));
    /*display_tree(population[0], 0);*/
    /*printf("\n");*/
    int index = 0;
    quicksort(results, population_count);
    printf("Score: %f\n", naive_average(results, population_count));
    float threshold = percentile(results, population_count, 0.9); // 90%
    printf("Low Quartile is: %f\n", threshold);
    /*printf("Best: %lf\n", results[0]);*/
    display_top(results, 10);

    t4 = get_timestamp();
    double t_secs = (t4 - t3) / 1000000.0L;

    selection(population, population_count, featuresPtr, DATASET_SIZE, results_cpy, threshold);
    // crossover(population, population_count, percentage);
    // mutate();
    // Crossover
    // Mutate individuals
    // Mutate
    free(results);
    free(results_cpy);
    printf("Tournament took: %.5f s\n", t_secs);
  }

  timestamp_t t2 = get_timestamp();
  double secs = (t2 - t1) / 1000000.0L;
  printf("Backtest took: %.3f s\n", secs);

  fclose(datasetFile);
}

int main(int argc, char **argv) {
  srand(time(NULL));

  FILE *logFile = fopen("./debug.json", "w");

  test(logFile);
  fclose(logFile);
  return 0;
}

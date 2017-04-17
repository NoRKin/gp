#ifndef CONSTANTS_H_

#define DEBUG 0
#define USE_CUDA false
#define NUMTHREADS 8

#define DATASET_SIZE 80000
#define FEATURE_COUNT 21

#define CROSSOVER_RATE 0.5
#define MUTATION_RATE 0.5

#define MAX_DEPTH 4
#define TREE_SIZE 256
#define GENERATIONS 300

#define THREADS 64
#define BLOCKS 64
#define POPULATION_SIZE THREADS * BLOCKS

#endif /* ifndef CONSTANTS_H_ */

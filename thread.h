#include "node.h"

#ifndef GPTHREAD_H_
#define GPTHREAD_H_


typedef struct s_gpthread {
  node **population;
  int  population_count;
  float const **features;
  float *results;
  int start;
  int end;
} gpthread;

#endif

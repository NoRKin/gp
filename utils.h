#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include "thread.h"

#include "node.h"
#include "operation.h"

#ifndef UTILS_H_
#define UTILS_H_

typedef unsigned long long timestamp_t;

//static  timestamp_t get_timestamp();
void    quicksort(float *A, int len);
void    display_top(float *results, int results_size, int n);
void    display_rpn(node *rpn);
void    display_feature_line(const float *line, int feature_count);

float   naive_average(float *a, int n);
float   percentile(float *results, int length, float top);
float   max(float a, float b);
float   min(float a, float b);
int     maxIndex(float *results, int a, int b);
int     minIndex(float *results, int a, int b);

int     rmdup(int *array, int length);
float   *concat(gpthread **gp, float * results, int pop_size, int thread_count);

bool    contains(int *array, int number);
void    uniq_rand(int *numbers, int max, int min);

#endif

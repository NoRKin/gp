#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#ifndef UTILS_H_
#define UTILS_H_

typedef unsigned long long timestamp_t;

//static  timestamp_t get_timestamp();
void    quicksort(float *A, int len);
void    display_top(float *results, int n);

float   naive_average(float *a, int n);
float   percentile(float *results, int length, float top);
float   max(float a, float b);
float   min(float a, float b);
int     maxIndex(float *results, int a, int b);
int     minIndex(float *results, int a, int b);

bool    contains(int *array, int number);
void    uniq_rand(int *numbers, int max, int min);

#endif

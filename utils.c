#include "utils.h"

/*static timestamp_t get_timestamp() {*/
  /*struct timeval now;*/
  /*gettimeofday (&now, NULL);*/

  /*return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;*/
/*}*/

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

float percentile(float *results, int length, float top) {
  int portionIndex = floor(length * top);

  return results[portionIndex];
}

void display_top(float *results, int n) {
  for (int i = 0; i < n; i++) {
    printf("Position: %d\t Score: %f\n", i, results[i]);
  }
}

void display_rpn(node *rpn) {
  int i = 0;

  while (rpn[i].operation != -2) {
    if (rpn[i].operation != -1) {
      printf(" %c", operation_label(rpn[i].operation));
    }
    else {
      printf(" %d", rpn[i].feature);
    }
    i++;
  }
  printf("\n");
}

void display_feature_line(const float *line, int feature_count) {
  int i = 0;

  for (i = 0; i < feature_count; i++) {
    printf(" %f,", line[i]);
  }
  printf("\n");
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

int maxIndex(float *results, int a, int b) {
  if (results[a] > results[b])
    return a;
  else
    return b;
}

int minIndex(float *results, int a, int b) {
  if (results[a] < results[b])
    return a;
  else
    return b;
}

bool contains(int *array, int number) {
  int i = 0;
  for (i = 0; i < 4; i++) {
    if (array[i] == number)
      return true;
  }

  return false;
}


void uniq_rand(int *numbers, int max, int min) {
  int i = 0;
  int value = 0;
  /*printf("UNIQU RAND Val is %d\n", value);*/
  for (i = 0; i < 4; i++) {
    while (contains(numbers, value)) {
      value = (rand() % max) + min;
    }

    /*if (value > max)*/
      /*value = max;*/
    /*printf("UNIQU RAND Val is %d, %d\n", i, value);*/
    numbers[i] = value;
  }
}

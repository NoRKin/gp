#include "operation.h"


float operation_run(int operation, float a, float b) {
  if (operation == 0) {
    return a + b;
  }
  if (operation == 1) {
    return a * b;
  }
  if (operation == 2) {
    return a / b;
  }
  if (operation == 3) {
    return a - b;
  }
  if (operation == 4) {
    return sqrt(a);
  }
  if (operation == 5) {
    return fdim(a, b);
  }
  if (operation == 6) {
    return log(a);
  }
  if (operation == 7) {
    return exp(a);
  }
  if (operation == 8) {
    return cos(a);
  }
  if (operation == 9) {
    return sin(a);
  }

  return 0;
}

char operation_label(int operation) {
  if (operation == 0) {
    return '+';
  }
  if (operation == 1) {
    return '.';
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

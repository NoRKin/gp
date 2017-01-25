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

  return '!';
}

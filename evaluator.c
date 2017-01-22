#include "evaluator.h"

float evaluate_tree(node *memspace, int offset, const float *features) {
  // If operation node
  if (memspace[offset].operation != -1) {
    float left = evaluate_tree(memspace, offset * 2 + 1, features);
    float right = evaluate_tree(memspace, offset * 2 + 2, features);

    return operation_run(memspace[offset].operation, left, right);
  }
  else {
    // Term node
    return features[memspace[offset].feature];
  }
}

void  display_tree(node *memspace, int offset) {
  if (memspace[offset].operation != -1) {
    // Left
    printf("(");
    display_tree(memspace, offset * 2 + 1);

    printf("%c", operation_label(memspace[offset].operation));
    // Right
    display_tree(memspace, offset * 2 + 2);
    printf(")");

    /*printf("(_%f)", features[memspace[offset].feature);*/
    /*return operation_run(memspace[offset].operation, left, right);*/
  }
  else {
    // Term node
    printf("_%d", memspace[offset].feature);
  }
}


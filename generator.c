#include "generator.h"

void generate_tree(node *memspace, int offset, int depth, int max_depth, int feature_count) {
  /*printf("Generate tree with offset %d and depth %d, %d\n", offset, depth, max_depth);*/
  if (depth < max_depth) {
    memspace[offset].operation = rand() % OPERATION_COUNT + OPERATION_COUNT;

    // If operation node
    if (memspace[offset].operation < OPERATION_COUNT) {
      // Right
      memspace[offset * 2 + 1].feature = rand() % feature_count;
      memspace[offset * 2 + 1].operation = -1;

      // Left
      memspace[offset * 2 + 2].feature = rand() % feature_count;
      memspace[offset * 2 + 2].operation = -1;
    }
    else {
      memspace[offset].operation -= OPERATION_COUNT;

      // Right
      generate_tree(memspace, offset * 2 + 1, depth + 1, max_depth, feature_count);

      // Left
      generate_tree(memspace, offset * 2 + 2, depth + 1, max_depth, feature_count);
    }
  }
  else {
    // Right
    memspace[offset * 2 + 1].feature = rand() % feature_count;
    memspace[offset * 2 + 1].operation = -1;

    // Left
    memspace[offset * 2 + 2].feature = rand() % feature_count;
    memspace[offset * 2 + 2].operation = -1;
  }
}

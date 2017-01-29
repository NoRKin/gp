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

float fast_evaluate_tree(node *memspace, int offset, const float *features) {

  if (memspace[offset].operation == -1) {
    return features[memspace[offset].feature];
  }
  else {
    float left = evaluate_tree(memspace, offset * 2 + 1, features);
    float right = evaluate_tree(memspace, offset * 2 + 2, features);

    return operation_run(memspace[offset].operation, left, right);
  }
}

int tree_depth(node *memspace, int offset, int depth) {
  if (memspace[offset].operation != -1) {
    int left = tree_depth(memspace, offset * 2 + 1, depth + 1);
    int right = tree_depth(memspace, offset * 2 + 2, depth + 1);

    if (left > right)
      return left;
    else
      return right;
  }
  else {
    // Term node
    return depth;
  }
}

// Return Random subtree offset
int random_subtree(node *memspace, int offset, int min_offset, int chances) {
  // 40 chance
  //
  int ret = rand() % 100;
  if (ret > chances && offset > min_offset) {
    return offset;
  }

  if (memspace[offset].operation != -1) {
    // Random left right
    int left_or_right = rand() % 2 + 1;


    // Do not return offset > 16
    return random_subtree(memspace, offset * 2 + left_or_right, min_offset, chances - 10);
  }
  else {
    if (offset < min_offset) {
      printf("ERROR: Terminal node %d\n", offset);
    }
    return offset;
  }
}

int tree_nodes_count(node *memspace, int offset, int count) {
  if (memspace[offset].operation != -1) {
    int left = tree_depth(memspace, offset * 2 + 1, count + 1);
    int right = tree_depth(memspace, offset * 2 + 2, count + 1);

    return left + right;
  }
  else {
    // Term node
    return count + 1;
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

void mutate_tree(node **population, int from, int offset, int chances) {

  int ret = rand() % 100;
  if (ret > chances) {
    if (population[from][offset].operation != -1) {
      population[from][offset].operation = rand() % OPERATION_COUNT;
    }
    else {
      population[from][offset].feature = rand() % 21; // FEATURE_COUNT
    }
    // Mutate
    return ;
  }

  if (population[from][offset].operation != -1) {
    int left_or_right = rand() % 2 + 1;

    mutate_tree(population, from, offset + 2 + left_or_right, chances - 10);
  }
  else {
    // Terminal .. mutate();
    population[from][offset].feature = rand() % 50; // FEATURE_COUNT
  }
}


void copy_branch(node **population, int from, int to, int offset_from, int offset_to) {
  // copy
  /*memcpy(to, from, sizeof(node));*/
  /*printf("Copy operation %d\n", population[from][offset_from].operation);*/
  if (offset_to > 128)
    printf("Offsets are from %d, to %d\n", offset_from, offset_to);

  population[to][offset_to].operation = population[from][offset_from].operation;
  /*puts("Copy feature");*/
  population[to][offset_to].feature = population[from][offset_from].feature;
  /*puts("Copy done");*/

  if (population[from][offset_from].operation != -1) {
    if (offset_to * 2 + 2 < 128) {
      copy_branch(population, from, to, offset_from * 2 + 1, offset_to * 2 + 1);
      copy_branch(population, from, to, offset_from * 2 + 2, offset_to * 2 + 2);
    }
    else {
      // TOO LONG, we replace it with a terminal node (we assume mutation at the same time)
      population[to][offset_to].feature = rand() % 50; // FEATURE_COUNT
      population[to][offset_to].operation = -1;
    }

    return ;
  }
  else {
    // Term node
    return ;
  }
}

void tree_to_rpn(node *memspace, int offset, node *to, int *to_offset) {
  if (memspace[offset].operation != -1) {
    // Left
    /*printf("(");*/
    tree_to_rpn(memspace, offset * 2 + 1, to, to_offset);

    /*printf("%c", operation_label(memspace[offset].operation));*/
    // Right
    tree_to_rpn(memspace, offset * 2 + 2, to, to_offset);
    /*printf(")");*/
    // stack_push
    to[*to_offset].operation = memspace[offset].operation;
    (*to_offset)++;
    /*printf("(_%f)", features[memspace[offset].feature);*/
    /*return operation_run(memspace[offset].operation, left, right);*/
  }
  else {
    // Term node
    to[*to_offset].feature = memspace[offset].feature;
    to[*to_offset].operation = -1;
    (*to_offset)++;
  }
}

float eval_rpn(node *rpn, const float *features) {
  float stack[16];
  int s_index = 0;
  int i = 0;

  while(rpn[i].operation != -2) {
    if (rpn[i].operation == -1) {
      stack[s_index] = features[rpn[i].feature];
      s_index++;
    }
    // Function, we want to reduce the stack
    else if (s_index > 1) {
      s_index = s_index - 1;
      stack[s_index - 1] = operation_run(rpn[i].operation, stack[s_index - 1], stack[s_index]);
    }
    // Stack is not full enough
    /*else {*/
    /*printf("Error the stack is not full enough %d\n", s_index);*/
    /*}*/
    i++;
  }

  return stack[0];
}

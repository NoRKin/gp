#include "constants.h"
#include "node.h"
#include "operation.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifndef EVALUATOR_H_
#define EVALUATOR_H_

float evaluate_tree(node *memspace, int offset, const float *features);
float fast_evaluate_tree(node *memspace, int offset, const float *features);
void  display_tree(node *memspace, int offset);
void  copy_branch(node **population, int from, int to, int offsetFrom, int offsetTo);
void  copy_tree(node *from, node *to, int offset);
int   tree_depth(node *memspace, int offset, int depth);
int   tree_nodes_count(node *memspace, int offset, int count);
int   random_subtree(node *memspace, int offset, int min_offset, int chances);
void  mutate_tree(node **population, int from, int offset, int chances);
void  tree_to_rpn(node *memspace, int offset, node *to, int *to_offset);
float eval_rpn(node *rpn, const float *features);

#endif

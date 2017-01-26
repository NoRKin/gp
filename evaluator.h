#include "node.h"
#include "operation.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifndef EVALUATOR_H_
#define EVALUATOR_H_

float evaluate_tree(node *memspace, int offset, const float *features);
void  display_tree(node *memspace, int offset);
void  copy_branch(node **population, int from, int to, int offsetFrom, int offsetTo);
int   tree_depth(node *memspace, int offset, int depth);
int   tree_nodes_count(node *memspace, int offset, int count);
int   random_subtree(node *memspace, int offset, int min_offset, int chances);
void  mutate_tree(node **population, int from, int offset, int chances);

#endif

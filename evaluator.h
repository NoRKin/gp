#include "node.h"
#include "operation.h"
#include <stdio.h>

#ifndef EVALUATOR_H_
#define EVALUATOR_H_

float evaluate_tree(node *memspace, int offset, const float *features);
void  display_tree(node *memspace, int offset);

#endif

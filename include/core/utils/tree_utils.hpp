#pragma once

#include "core/tensor.hpp"

/*
Gini impurity for a set of class labels (multiclass).

Gini impurity = 1 - Σ(p_i²)
where p_i is the proportion of samples belonging to class i in the node.

Gini impurity measures how mixed the classes are in a node.

0.0 means that the node is pure (all the training samples in this node belong to one class).
Higher values mean that the node contains a variety of classes.
*/
double gini_impurity(const Tensor &y);

double majority_class(const Tensor &y);
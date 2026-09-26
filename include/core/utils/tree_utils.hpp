#pragma once

#include "core/tensor.hpp"

#include <vector>
#include <random>

namespace cpp_ml {

bool all_same_class(const Tensor &y);

/*
Takes X and y input data and splits into left/right groups based on
X[row, feature_index] <= threshold for left and X[row, feature_index] > threshold
*/
void split_dataset(
    const Tensor &X,
    const Tensor &y,
    int feature_index,
    double threshold,
    Tensor &X_left,
    Tensor &y_left,
    Tensor &X_right,
    Tensor &y_right
);

/*
sample_feature_indices returns a vector of the indices of features chosen to be
considered for creating splits in nodes
*/
std::vector<int> sample_feature_indices(
    int num_features,
    int max_features,
    std::mt19937 &generator
);

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

}

#pragma once

#include "cpp_ml/core/tensor.hpp"

#include <cstddef>
#include <random>
#include <vector>

namespace cpp_ml {

/**
 * @brief returns true if all the values in a column vector are the same
 * 
 * @param y is a column vector of the classes
 */
bool all_same_class(const Tensor &y);

/**
 * @brief Splits a dataset into 2 on the feature index and threshold
 * that minimizes the weighted Gini Impurity the most.
 * 
 * @param feature_index Which feature in X to split on
 * @param threshold The threshold of the feature to split on
 */
void split_dataset(
    const Tensor &X,
    const Tensor &y,
    std::size_t feature_index,
    double threshold,
    Tensor &X_left,
    Tensor &y_left,
    Tensor &X_right,
    Tensor &y_right
);

/**
 * @brief Randomly generates feature indices. Used in Random Forests.
 * The features indices are positive integers starting from 0.
 * 
 * @param num_features Number of unique features to generate
 * @param max_features Total number of features available to chose from.
 * @return std::vector<std::size_t> Vector of the feature_indices chosen
 */
std::vector<std::size_t> sample_feature_indices(
    std::size_t num_features,
    std::size_t max_features,
    std::mt19937 &generator
);

/**
 * @brief Gini impurity for a set of class labels (multiclass).
 * 
 * Gini impurity = 1 - Σ(p_i²)
 * where p_i is the proportion of samples belonging to class i in the node.
 * 
 * Gini impurity measures how mixed the classes are in a node.
 * 0.0 means that the node is pure (all the training samples in this node belong to one class).
 * Higher values mean that the node contains a variety of classes.
 * 
 * @param y Column vector of the classes
 */
double gini_impurity(const Tensor &y);

/**
 * @brief Returns the most common value in a column vector
 */
double majority_class(const Tensor &y);

// Constructor validation
std::size_t validate_max_depth(std::size_t max_depth);
std::size_t validate_min_samples_split(std::size_t min_samples_split);
std::size_t validate_max_features(std::size_t max_features);

}

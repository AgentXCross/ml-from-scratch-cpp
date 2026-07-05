#include "models/random_forest.hpp"

#include <map>
#include <random>
#include <stdexcept>
#include <vector>

RandomForest::RandomForest() {
    num_trees_ = 10;
    max_depth_ = 5;
    min_samples_split_ = 2;
    max_features_ = 0;

    random_seed_ = 42;
    fitted_ = false;
}

RandomForest::RandomForest(
    int num_trees,
    int max_depth,
    int min_samples_split,
    int max_features,
    unsigned int random_seed
) {
    if (num_trees <= 0) {
        throw std::invalid_argument("num_trees must be positive");
    }

    if (max_depth <= 0) {
        throw std::invalid_argument("max_depth must be positive");
    }

    if (min_samples_split <= 1) {
        throw std::invalid_argument("min_samples_split must be greater than 1");
    } 

    if (max_features <= 0) {
        throw std::invalid_argument("max_features must be positive");
    }

    num_trees_ = num_trees;
    max_depth_ = max_depth;
    min_samples_split_ = min_samples_split;
    max_features_ = max_features;

    random_seed_ = random_seed;
    fitted_ = false;
}

void RandomForest::fit(
    const Matrix &X,
    const Matrix &y
) {
    
}
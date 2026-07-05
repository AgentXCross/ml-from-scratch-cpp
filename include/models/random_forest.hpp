#pragma once

#include "core/matrix.hpp"
#include "models/decision_tree.hpp"

#include <memory>
#include <vector>

class RandomForest {
private:
    // owns each DecisionTree through a unique_ptr
    std::vector<std::unique_ptr<DecisionTree>> trees_; 

    int num_trees_;
    int max_depth_;
    int min_samples_split_;
    int max_features_;

    unsigned int random_seed_;
    bool fitted_;

public:
    RandomForest();

    RandomForest(
        int num_trees,
        int max_depth,
        int min_samples_split,
        int max_features,
        unsigned int random_seed = 42
    );

    void fit(
        const Matrix &X,
        const Matrix &y
    );

    Matrix predict(const Matrix &X) const;

    int num_trees() const;
};
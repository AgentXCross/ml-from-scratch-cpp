#pragma once

#include "core/tensor.hpp"
#include "models/decision_tree_classifier.hpp"

#include <memory>
#include <vector>

class RandomForestClassifier {
private:
    // owns each DecisionTreeClassifier through a unique_ptr
    std::vector<std::unique_ptr<DecisionTreeClassifier>> trees_; 

    int num_trees_;
    int max_depth_;
    int min_samples_split_;
    int max_features_;

    unsigned int random_seed_;
    bool fitted_;

public:
    RandomForestClassifier();

    RandomForestClassifier(
        int num_trees,
        int max_depth,
        int min_samples_split,
        int max_features,
        unsigned int random_seed = 42
    );

    void fit(
        const Tensor &X,
        const Tensor &y
    );

    Tensor predict(const Tensor &X) const;

    int num_trees() const;
};

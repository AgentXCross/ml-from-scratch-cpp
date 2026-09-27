#pragma once

#include "cpp_ml/core/tensor.hpp"

#include <cstddef>
#include <memory>
#include <random>

namespace cpp_ml {

struct DecisionTreeClassifierNode {
    bool is_leaf; // A leaf is a terminal node in a decision tree that makes the prediction

    int feature_index; // Which feature this node considers
    double threshold; // Threshold value of the feature
    double prediction; // Prediction value if this node is a leaf

    std::unique_ptr<DecisionTreeClassifierNode> left;
    std::unique_ptr<DecisionTreeClassifierNode> right;

    DecisionTreeClassifierNode();
};

class DecisionTreeClassifier {
private:
    std::unique_ptr<DecisionTreeClassifierNode> root_;

    int max_depth_;
    int min_samples_split_;
    int max_features_;
    unsigned int random_seed_;

    std::mt19937 generator_;

    bool fitted_;

    std::unique_ptr<DecisionTreeClassifierNode> build_tree(
        const Tensor &X,
        const Tensor &y,
        int depth // stores the current depth as we recurse
    );

    double predict_sample(
        const Tensor &x,
        const DecisionTreeClassifierNode *node
    ) const;

public:
    DecisionTreeClassifier();
    DecisionTreeClassifier(
        int max_depth, 
        int min_samples_split,
        int max_features = 0, // 0 means to use all features
        unsigned int random_seed = 42
    );

    void fit(
        const Tensor &X,
        const Tensor &y
    );

    Tensor predict(const Tensor &X) const;
};

}

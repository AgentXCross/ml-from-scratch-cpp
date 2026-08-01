#pragma once

#include "core/tensor.hpp"

#include <memory>
#include <random>

struct DecisionTreeNode {
    bool is_leaf; // A leaf is a terminal node in a decision tree that makes the prediction

    int feature_index; // Which feature this node considers
    double threshold; // Threshold value of the feature
    double prediction; // Prediction value if this node is a leaf

    std::unique_ptr<DecisionTreeNode> left;
    std::unique_ptr<DecisionTreeNode> right;

    DecisionTreeNode();
};

class DecisionTreeClassifier {
private:
    std::unique_ptr<DecisionTreeNode> root_;

    int max_depth_;
    int min_samples_split_;
    int max_features_;
    unsigned int random_seed_;

    std::mt19937 generator_;

    bool fitted_;

    /*
    build_tree 
    */
    std::unique_ptr<DecisionTreeNode> build_tree(
        const Tensor &X,
        const Tensor &y,
        int depth
    );

    double predict_sample(
        const Tensor &x,
        const DecisionTreeNode *node
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

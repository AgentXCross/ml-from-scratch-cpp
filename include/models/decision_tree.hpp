#pragma once

#include "core/matrix.hpp"

#include <memory>
#include <random>

struct DecisionTreeNode {
    bool is_leaf; // A leaf is a terminal node in a decision tree that makes the prediction

    int feature_index;
    double threshold;
    double prediction;

    std::unique_ptr<DecisionTreeNode> left;
    std::unique_ptr<DecisionTreeNode> right;

    DecisionTreeNode();
};

class DecisionTree {
private:
    std::unique_ptr<DecisionTreeNode> root_;

    int max_depth_;
    int min_samples_split_;
    int max_features_;
    unsigned int random_seed_;

    std::mt19937 generator_;

    bool fitted_;

    std::unique_ptr<DecisionTreeNode> build_tree(
        const Matrix &X,
        const Matrix &y,
        int depth
    );

    double predict_sample(
        const Matrix &x,
        const DecisionTreeNode *node
    ) const;

public:
    DecisionTree();
    DecisionTree(
        int max_depth, 
        int min_samples_split,
        int max_features = 0, // 0 means to use all features
        unsigned int random_seed = 42
    );

    void fit(
        const Matrix &X,
        const Matrix &y
    );

    Matrix predict(const Matrix &X) const;
};
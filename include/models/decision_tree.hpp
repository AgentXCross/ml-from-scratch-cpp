#pragma once

#include "core/matrix.hpp"

struct DecisionTreeNode {
    bool is_leaf; // A leaf is a terminal node in a decision tree that makes the prediction

    int feature_index;
    double threshold;

    double prediction;

    DecisionTreeNode *left;
    DecisionTreeNode *right;

    DecisionTreeNode();
};

class DecisionTree {
private:
    DecisionTreeNode *root_;

    int max_depth_;
    int min_samples_split_;

    DecisionTreeNode *build_tree(
        const Matrix &X,
        const Matrix &y,
        int depth
    );

    double predict_sample(
        const Matrix &x,
        const DecisionTreeNode *node
    ) const;

    void free_tree(DecisionTreeNode *node);

public:
    DecisionTree();
    DecisionTree(int max_depth, int min_samples_split);

    ~DecisionTree();

    void fit(
        const Matrix &X,
        const Matrix &y
    );

    Matrix predict(const Matrix &X) const;
};
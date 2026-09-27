#pragma once

#include "cpp_ml/core/tensor.hpp"

#include <memory>
#include <random>

namespace cpp_ml {

struct DecisionTreeRegressorNode {
    bool is_leaf; // leaf nodes store the Decision Tree's final value

    int feature_index; // feature index considered by this node
    double threshold; // threshold on the feature
    double prediction; // prediction value if this node is a leaf

    std::unique_ptr<DecisionTreeRegressorNode> left;
    std::unique_ptr<DecisionTreeRegressorNode> right;

    DecisionTreeRegressorNode();
};

class DecisionTreeRegressor {
private:
    std::unique_ptr<DecisionTreeRegressorNode> root_;

    int max_depth_;
    int min_samples_split_;
    int max_features_;
    unsigned int random_seed_;

    std::mt19937 generator_;

    bool fitted_;

    std::unique_ptr<DecisionTreeRegressorNode> build_tree(
        const Tensor& X,
        const Tensor& y,
        int depth
    );

    double predict_sample(
        const Tensor& x,
        const DecisionTreeRegressorNode* node
    );

public:
    DecisionTreeRegressor();
    DecisionTreeRegressor(
        int max_depth,
        int min_samples_split,
        int max_features = 0, // 0 means use all the features
        unsigned int random_seed = 42
    );

    void fit(
        const Tensor &X,
        const Tensor &y
    );

    Tensor predict(const Tensor &X) const;
};

}

#pragma once

#include "cpp_ml/core/tensor.hpp"

#include <cstddef>
#include <memory>
#include <random>

namespace cpp_ml {

struct DecisionTreeRegressorNode {
    bool is_leaf; // leaf/terminal nodes store the Decision Tree's final value

    std::size_t feature_index; // feature index considered by this node
    double threshold; // threshold on the feature
    double prediction; // prediction value if this node is a leaf

    std::unique_ptr<DecisionTreeRegressorNode> left;
    std::unique_ptr<DecisionTreeRegressorNode> right;

    DecisionTreeRegressorNode();

};

class DecisionTreeRegressor {
private:
    std::unique_ptr<DecisionTreeRegressorNode> root_;

    std::size_t max_depth_;
    std::size_t min_samples_split_; // min samples in a node required to make a split
    std::size_t max_features_; // 
    std::size_t random_seed_;

    std::mt19937 generator_;

    bool fitted_;

    /**
     * @brief Recursively create nodes, returns the root.
     * 
     * create node
     * if stopping condition 
     * (impurity is 0.0 or reached max depth or not enough samples for split or no split found):
     *   make leaf predicting the avg of the samples in this node
     * else:
     *   find best split
     *   split data
     *   recursively build the left and right children
     */
    std::unique_ptr<DecisionTreeRegressorNode> build_tree(
        const Tensor &X,
        const Tensor &y,
        std::size_t depth
    );

    double predict_sample(
        const Tensor &x,
        const DecisionTreeRegressorNode *node
    );

public:
    DecisionTreeRegressor(); // Delegating constructor
    DecisionTreeRegressor(
        std::size_t max_depth,
        std::size_t min_samples_split,
        std::size_t max_features = 0, // 0 means use all the features
        std::size_t random_seed = 42
    );

    void fit(
        const Tensor &X,
        const Tensor &y
    );

    Tensor predict(const Tensor &X) const;
};

}

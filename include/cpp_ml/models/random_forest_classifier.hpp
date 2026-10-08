#pragma once

#include "cpp_ml/core/tensor.hpp"
#include "cpp_ml/models/decision_tree_classifier.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace cpp_ml {

class RandomForestClassifier {
private:
    // owns each DecisionTreeClassifier through a unique_ptr
    std::vector<std::unique_ptr<DecisionTreeClassifier>> trees_; 

    std::size_t num_trees_;
    std::size_t max_depth_;
    std::size_t min_samples_split_;
    std::size_t max_features_;

    unsigned int random_seed_;
    bool fitted_;

public:
    RandomForestClassifier();

    RandomForestClassifier(
        std::size_t num_trees,
        std::size_t max_depth,
        std::size_t min_samples_split,
        std::size_t max_features,
        unsigned int random_seed = 42
    );

    void fit(
        const Tensor &X,
        const Tensor &y
    );

    Tensor predict(const Tensor &X) const;

    std::size_t num_trees() const;
};

}

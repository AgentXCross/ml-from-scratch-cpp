#include "cpp_ml/models/decision_tree_regressor.hpp"

#include "cpp_ml/core/utils/tree_utils.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>

namespace cpp_ml {

DecisionTreeRegressorNode::DecisionTreeRegressorNode()
    : is_leaf{false},
      feature_index{0},
      threshold{0.0},
      prediction{0.0},
      left{nullptr},
      right{nullptr} {}

DecisionTreeRegressor::DecisionTreeRegressor()
    : DecisionTreeRegressor(5, 2, 0, 42) {}


DecisionTreeRegressor::DecisionTreeRegressor(
    std::size_t max_depth,
    std::size_t min_samples_split,
    std::size_t max_features,
    std::size_t random_seed
) 
    : root_{nullptr},
      max_depth_{validate_max_features(max_depth)},
      min_samples_split_{validate_min_samples_split(min_samples_split)},
      max_features_{validate_max_features(max_features)},
      random_seed_{random_seed},
      generator_{static_cast<std::mt19937::result_type>(random_seed_)},
      fitted_{false} {}
}

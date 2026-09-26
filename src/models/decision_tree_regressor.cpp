#include "models/decision_tree_regressor.hpp"

namespace cpp_ml {

DecisionTreeNode::DecisionTreeNode()
    : is_leaf(false),
      feature_index(-1),
      threshold(0.0),
      prediction(0.0),
      left(nullptr),
      right(nullptr) {}

}

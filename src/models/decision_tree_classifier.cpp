#include "models/decision_tree_classifier.hpp"

#include "core/utils/tree_utils.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>

namespace cpp_ml {

/*
Weighted Gini Impurity = (n_left / n_total) * gini(left) + (n_right / n_total) * gini(right)
During training, we choose the split that has the lowest weighted gini impurity.
*/
double weighted_gini_impurity(
    const Tensor &y_left,
    const Tensor &y_right
) {
    if (!y_left.is_matrix() || !y_right.is_matrix()) {
        throw std::invalid_argument("y_left and y_right must be rank-2 tensors");
    }

    if (y_left.cols() != 1 || y_right.cols() != 1) {
        throw std::invalid_argument("y_left and y_right must both be column vectors");
    }

    int n_left = y_left.rows();
    int n_right = y_right.rows();
    int n_total = n_left + n_right;

    if (n_left == 0 || n_right == 0) {
        return 1.0; // invalid split to have one side have no data
    }

    double left_weight = static_cast<double> (n_left) / n_total;
    double right_weight = static_cast<double> (n_right) / n_total;

    return left_weight * gini_impurity(y_left)
            + right_weight * gini_impurity(y_right);
}


/*
find_best_split returns True if a valid split is found, false otherwise.
Method:
for each feature in the sampled features:
    for each value in that feature:
        try splitting there (split at the values of the training samples)
        score the split
        keep the best one
*/
bool find_best_split(
    const Tensor &X,
    const Tensor &y,
    int max_features,
    std::mt19937 &generator,
    int &best_feature_index,
    double &best_threshold
) {
    if (!X.is_matrix() || !y.is_matrix()) {
        throw std::invalid_argument("X and y must be rank-2 tensors");
    }

    if (X.rows() != y.rows()) {
        throw std::invalid_argument("X and y must have the same number of rows");
    }

    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("X cannot be empty");
    }

    if (y.cols() != 1) {
        throw std::invalid_argument("y must be a column vector");
    }

    double best_impurity = gini_impurity(y);
    bool found_split = false;

    best_feature_index = -1;
    best_threshold = 0.0;

    std::vector<int> feature_indices = sample_feature_indices(
        X.cols(),
        max_features,
        generator
    );

    for (int feature_index : feature_indices) {
        for (int i = 0; i < X.rows(); i++) {
            double threshold = X.at(i, feature_index);

            int n_left = 0;
            int n_right = 0;

            for (int sample_index = 0; sample_index < X.rows(); sample_index++) {
                if (X.at(sample_index, feature_index) <= threshold) {
                    n_left++;
                } else {
                    n_right++;
                }
            }

            if (n_left == 0 || n_right == 0) {
                continue;
            }

            Tensor X_left;
            Tensor y_left;
            Tensor X_right;
            Tensor y_right;

            split_dataset(
                X,
                y,
                feature_index,
                threshold,
                X_left,
                y_left,
                X_right,
                y_right
            );

            assert(X_left.rows() > 0);
            assert(X_right.rows() > 0);
            assert(y_left.rows() > 0);
            assert(y_right.rows() > 0);

            double impurity = weighted_gini_impurity(y_left, y_right);

            if (impurity < best_impurity) {
                best_impurity = impurity;
                best_feature_index = feature_index;
                best_threshold = threshold;
                found_split = true;
            }
        }
    }

    return found_split;
}


DecisionTreeNode::DecisionTreeNode()
    : is_leaf(false),
      feature_index(-1),
      threshold(0.0),
      prediction(0.0),
      left(nullptr),
      right(nullptr) {}


DecisionTreeClassifier::DecisionTreeClassifier()
    : root_(nullptr),
      max_depth_(5),
      min_samples_split_(2),
      max_features_(0),
      random_seed_(42),
      generator_(std::mt19937(random_seed_)),
      fitted_(false) {}


static int validate_max_depth(int max_depth) {
    if (max_depth <= 0) {
        throw std::invalid_argument("max_depth must be positive");
    }

    return max_depth;
}


static int validate_min_samples_split(int min_samples_split) {
    if (min_samples_split <= 1) {
        throw std::invalid_argument("min_samples_split must be greater than 1");
    }

    return min_samples_split;
}


static int validate_max_features(int max_features) {
    if (max_features < 0) {
        throw std::invalid_argument("max_features cannot be negative");
    }

    return max_features;
}


DecisionTreeClassifier::DecisionTreeClassifier(
    int max_depth, 
    int min_samples_split,
    int max_features,
    unsigned int random_seed
)
    : root_(nullptr),
      max_depth_(validate_max_depth(max_depth)),
      min_samples_split_(validate_min_samples_split(min_samples_split)),
      max_features_(validate_max_features(max_features)),
      random_seed_(random_seed),
      generator_(std::mt19937(random_seed_)),
      fitted_(false) {}


void DecisionTreeClassifier::fit(
    const Tensor &X,
    const Tensor &y
) {
    if (!X.is_matrix() || !y.is_matrix()) {
        throw std::invalid_argument("X and y must be rank-2 tensors");
    }

    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("X cannot be empty");
    }

    if (y.rows() == 0 || y.cols() != 1) {
        throw std::invalid_argument("y must be a non-empty column vector");
    }

    if (X.rows() != y.rows()) {
        throw std::invalid_argument("X and y must have the same number of rows");
    }

    root_ = build_tree(X, y, 0);
    assert(root_ != nullptr);
    fitted_ = true;
}


/*
Recursively create nodes, returns the root.

create node
if stopping condition 
(all same class or reached max depth or not enough samples for split or no split found):
    make leaf
else:
    find best split
    split data
    recursively build the left and right children
*/
std::unique_ptr<DecisionTreeNode> DecisionTreeClassifier::build_tree(
    const Tensor &X,
    const Tensor &y,
    int depth
) {
    if (!X.is_matrix() || !y.is_matrix()) {
        throw std::invalid_argument("X and y must be rank-2 tensors");
    }

    if (X.rows() != y.rows()) {
        throw std::invalid_argument("X and y must have the same number of rows");
    }

    if (y.cols() != 1) {
        throw std::invalid_argument("y must have exactly one column");
    }

    auto node = std::make_unique<DecisionTreeNode>();

    if (all_same_class(y) ||
        depth >= max_depth_ ||
        X.rows() < min_samples_split_
    ) {
        node->is_leaf = true;
        node->prediction = majority_class(y);

        return node;
    }

    int best_feature_index = -1;
    double best_threshold = 0.0;

    bool found_split = find_best_split(
        X,
        y,
        max_features_,
        generator_,
        best_feature_index,
        best_threshold
    );

    if (!found_split) {
        node->is_leaf = true;
        node->prediction = majority_class(y);

        return node;
    }

    Tensor X_left;
    Tensor y_left;
    Tensor X_right;
    Tensor y_right;

    split_dataset(
        X,
        y,
        best_feature_index,
        best_threshold,
        X_left,
        y_left,
        X_right,
        y_right
    );

    node->is_leaf = false;
    node->feature_index = best_feature_index;
    node->threshold = best_threshold;

    node->left = build_tree(X_left, y_left, depth + 1);
    node->right = build_tree(X_right, y_right, depth + 1);
    assert(node->left != nullptr);
    assert(node->right != nullptr);

    return node;
}


double DecisionTreeClassifier::predict_sample(
    const Tensor &x,
    const DecisionTreeNode *node
) const {
    if (node == nullptr) {
        throw std::runtime_error("Cannot predict using an empty tree");
    }

    if (!x.is_matrix() || x.rows() != 1) {
        throw std::invalid_argument("x must be a single row rank-2 tensor");
    }

    if (node->is_leaf) {
        return node->prediction;
    }

    if (x.at(0, node->feature_index) <= node->threshold) {
        return predict_sample(x, node->left.get());
    } else {
        return predict_sample(x, node->right.get());
    }
}


Tensor DecisionTreeClassifier::predict(const Tensor &X) const {
    if (root_ == nullptr) {
        throw std::runtime_error("DecisionTreeClassifier must be fitted before calling predict");
    }

    if (!fitted_) {
        throw std::runtime_error("DecisionTreeClassifier must be fitted before calling predict");
    }

    if (!X.is_matrix()) {
        throw std::invalid_argument("X must be a rank-2 tensor");
    }

    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("X cannot be empty");
    }

    Tensor predictions({X.rows(), 1});

    for (int i = 0; i < X.rows(); i++) {
        Tensor sample = X.row(i);

        predictions.at(i, 0) = predict_sample(sample, root_.get());
    }

    assert(predictions.is_matrix());
    assert(predictions.rows() == X.rows());
    assert(predictions.cols() == 1);

    return predictions;
}

}

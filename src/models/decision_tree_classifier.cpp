#include "models/decision_tree_classifier.hpp"

#include "core/utils/tree_utils.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>


bool all_same_class(const Tensor &y) {
    if (y.empty()) {
        throw std::invalid_argument("y cannot be empty");
    }

    if (!y.is_matrix() || y.cols() != 1) {
        throw std::invalid_argument("y must be a non-empty column vector");
    }

    double first_label = y.at(0, 0);

    for (int i = 1; i < y.rows(); i++) {
        if (y.at(i, 0) != first_label) {
            return false;
        }
    }

    return true;
}


/*
Takes X and y input data and splits into left/right groups based on
X[row, feature_index] <= threshold for left and X[row, feature_index] > threshold
*/
void split_dataset(
    const Tensor &X,
    const Tensor &y,
    int feature_index,
    double threshold,
    Tensor &X_left,
    Tensor &y_left,
    Tensor &X_right,
    Tensor &y_right
) {
    if (!X.is_matrix() || !y.is_matrix()) {
        throw std::invalid_argument("X and y must be rank-2 tensors");
    }

    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("X cannot be empty");
    }

    if (y.rows() == 0 || y.cols() == 0) {
        throw std::invalid_argument("y cannot be empty");
    }

    if (X.rows() != y.rows()) {
        throw std::invalid_argument("X and y must have the same number of rows");
    }

    if (y.cols() != 1) {
        throw std::invalid_argument("y must have exactly one column");
    }

    if (feature_index < 0 || feature_index >= X.cols()) {
        throw std::invalid_argument("feature_index is out of index");
    }

    std::vector<std::vector<double>> X_left_values;
    std::vector<std::vector<double>> y_left_values;
    std::vector<std::vector<double>> X_right_values;
    std::vector<std::vector<double>> y_right_values;

    for (int i = 0; i < X.rows(); i++) {
        std::vector<double> X_row;

        for (int j = 0; j < X.cols(); j++) {
            X_row.push_back(X.at(i, j));
        }

        std::vector<double> y_row = {y.at(i, 0)};

        if (X.at(i, feature_index) <= threshold) {
            X_left_values.push_back(X_row);
            y_left_values.push_back(y_row);
        } else {
            X_right_values.push_back(X_row);
            y_right_values.push_back(y_row);
        }
    }

    if (X_left_values.empty() || X_right_values.empty()) {
        throw std::invalid_argument("Split must put at least one sample on each side");
    }

    X_left = Tensor::from_vector(X_left_values);
    y_left = Tensor::from_vector(y_left_values);
    X_right = Tensor::from_vector(X_right_values);
    y_right = Tensor::from_vector(y_right_values);

    assert(X_left.is_matrix());
    assert(y_left.is_matrix());
    assert(X_right.is_matrix());
    assert(y_right.is_matrix());
    assert(X_left.rows() == y_left.rows());
    assert(X_right.rows() == y_right.rows());
    assert(X_left.cols() == X.cols());
    assert(X_right.cols() == X.cols());
    assert(y_left.cols() == 1);
    assert(y_right.cols() == 1);
}


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
sample_feature_indices returns a vector of the indices of features chosen to be
considered for creating splits in nodes
*/
std::vector<int> sample_feature_indices(
    int num_features,
    int max_features,
    std::mt19937 &generator
) {
    if (num_features <= 0) {
        throw std::invalid_argument("num_features must be positive");
    }

    if (max_features < 0) {
        throw std::invalid_argument("max_features cannot be negative");
    }

    std::vector<int> feature_indices;

    for (int feature_index = 0; feature_index < num_features; feature_index++) {
        feature_indices.push_back(feature_index);
    }

    if (max_features == 0 || max_features >= num_features) {
        return feature_indices;
    }

    std::shuffle(
        feature_indices.begin(),
        feature_indices.end(),
        generator
    );

    feature_indices.resize(max_features);

    return feature_indices;
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

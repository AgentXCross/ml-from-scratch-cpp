#include "models/decision_tree.hpp"

#include "core/utils/tree_utils.hpp"

#include <stdexcept>
#include <vector>

bool all_same_class(const Matrix &y) {
    if (y.rows() == 0 || y.cols() != 1) {
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
    const Matrix &X,
    const Matrix &y,
    int feature_index,
    double threshold,
    Matrix &X_left,
    Matrix &y_left,
    Matrix &X_right,
    Matrix &y_right
) {
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

    X_left = Matrix::from_vector(X_left_values);
    y_left = Matrix::from_vector(y_left_values);
    X_right = Matrix::from_vector(X_right_values);
    y_right = Matrix::from_vector(y_right_values);
}


/*
Weighted Gini Impurity = (n_left / n_total) * gini(left) + (n_right / n_total) * gini(right)
During training, we choose the split that has the lowest weighted gini impurity.
*/
double weighted_gini_impurity(
    const Matrix &y_left,
    const Matrix &y_right
) {
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
for each feature:
    for each value in that feature:
        try splitting there (split at the values of the training samples)
        score the split
        keep the best one
*/
bool find_best_split(
    const Matrix &X,
    const Matrix &y,
    int &best_feature_index,
    double &best_threshold
) {
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

    for (int feature_index = 0; feature_index < X.cols(); feature_index++) {
        for (int i = 0; i < X.rows(); i++) {
            double threshold = X.at(i, feature_index);

            Matrix X_left;
            Matrix y_left;
            Matrix X_right;
            Matrix y_right;

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

            if (X_left.rows() == 0 || X_right.rows() == 0) {
                continue;
            }

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


DecisionTreeNode::DecisionTreeNode() {
    is_leaf = false;

    feature_index = -1;
    threshold = 0.0;

    prediction = 0.0;

    left = nullptr;
    right = nullptr;
}


DecisionTree::DecisionTree() {
    root_ = nullptr;

    max_depth_ = 5;
    min_samples_split_ = 2;
}


DecisionTree::DecisionTree(int max_depth, int min_samples_split) {
    if (max_depth <= 0) {
        throw std::invalid_argument("max_depth must be positive");
    }

    if (min_samples_split <= 1) {
        throw std::invalid_argument("min_samples_split must be greater than 1");
    }

    root_ = nullptr;

    max_depth_ = max_depth;
    min_samples_split_ = min_samples_split;
}


DecisionTree::~DecisionTree() {
    free_tree(root_);
}


void DecisionTree::free_tree(DecisionTreeNode *node) {
    if (node == nullptr) {
        return;
    }

    free_tree(node->left);
    free_tree(node->right);

    delete node;
}


void DecisionTree::fit(
    const Matrix &X,
    const Matrix &y
) {
    if (X.rows() == 0 || X.cols() == 0) {
        throw std::invalid_argument("X cannot be empty");
    }

    if (y.rows() == 0 || y.cols() != 1) {
        throw std::invalid_argument("y must be a non-empty column vector");
    }

    if (X.rows() != y.rows()) {
        throw std::invalid_argument("X and y must have the same number of rows");
    }

    free_tree(root_);

    root_ = build_tree(X, y, 0);
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
DecisionTreeNode *DecisionTree::build_tree(
    const Matrix &X,
    const Matrix &y,
    int depth
) {
    DecisionTreeNode *node = new DecisionTreeNode();

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
        best_feature_index,
        best_threshold
    );

    if (!found_split) {
        node->is_leaf = true;
        node->prediction = majority_class(y);

        return node;
    }

    Matrix X_left;
    Matrix y_left;
    Matrix X_right;
    Matrix y_right;

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

    return node;
}

double DecisionTree::predict_sample(
    const Matrix &x,
    const DecisionTreeNode *node
) const {
    if (node == nullptr) {
        throw std::runtime_error("Cannot predict using an empty tree");
    }

    if (node->is_leaf) {
        return node->prediction;
    }

    if (x.at(0, node->feature_index) <= node->threshold) {
        return predict_sample(x, node->left);
    } else {
        return predict_sample(x, node->right);
    }
}

Matrix DecisionTree::predict(const Matrix &X) const {
    if (root_ == nullptr) {
        throw std::runtime_error("DecisionTree must be fitted before calling predict");
    }

    Matrix predictions(X.rows(), 1);

    for (int i = 0; i < X.rows(); i++) {
        Matrix sample = X.row(i);

        predictions.at(i, 0) = predict_sample(sample, root_);
    }

    return predictions;
}
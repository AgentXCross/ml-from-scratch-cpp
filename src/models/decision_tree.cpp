#include "models/decision_tree.hpp"

#include <map>
#include <stdexcept>
#include <vector>


/*
Gini impurity for a set of class labels (multiclass).

Gini impurity = 1 - Σ(p_i²)
where p_i is the proportion of samples belonging to class i in the node.

Gini impurity measures how mixed the classes are in a node.

0.0 means that the node is pure (all the training samples in this node belong to one class).
Higher values mean that the node contains a variety of classes.
*/
double gini_impurity(const Matrix &y) {
    if (y.rows() == 0 || y.cols() != 1) {
        throw std::invalid_argument("y must be a non-empty column vector");
    }

    std::map<double, int> class_counts;

    for (int i = 0; i < y.rows(); i++) {
        double label = y.at(i, 0);
        class_counts[label]++;
    }

    double impurity = 1.0;

    for (const auto &[key, value] : class_counts) {
        int count = value;
        double probability = static_cast<double> (count) / y.rows();

        impurity = impurity - probability * probability;
    }

    return impurity;
}


double majority_class(const Matrix &y) {
    if (y.rows() == 0 || y.cols() != 1) {
        throw std::invalid_argument("y must be a non-empty column vector");
    }

    std::map<double, int> class_counts;

    for (int i = 0; i < y.rows(); i++) {
        double label = y.at(i, 0);
        class_counts[label]++;
    }

    double best_label = 0.0;
    int best_count = -1;

    for (const auto &[key, value] : class_counts) {
        double label = key;
        int count = value;

        if (count > best_count) {
            best_label = label;
        }
    }

    return best_label;
}


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
        try splitting there
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
        
    }
}


#include "models/decision_tree.hpp"

#include <map>
#include <stdexcept>


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
    std::vector<std::vec>
}
#include "cpp_ml/core/utils/tree_utils.hpp"

#include <algorithm>
#include <cassert>
#include <map>
#include <stdexcept>

namespace cpp_ml {

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


double gini_impurity(const Tensor &y) {
    if (y.empty()) {
        throw std::invalid_argument("y cannot be empty");
    }

    if (!y.is_matrix() || y.cols() != 1) {
        throw std::invalid_argument("y must be a non-empty column vector with shape {n_samples, 1}");
    }

    std::map<double, int> class_counts;

    for (int i = 0; i < y.rows(); i++) {
        class_counts[y.at(i, 0)]++;
    }
    
    double impurity = 1.0;

    for (const auto &[label, count] : class_counts) {
        double probability = static_cast<double> (count) / y.rows();
        impurity -= probability * probability;
    }

    return impurity;
}


double majority_class(const Tensor &y) {
    if (y.empty()) {
        throw std::invalid_argument("y cannot be empty");
    }

    if (!y.is_matrix() || y.cols() != 1) {
        throw std::invalid_argument("y must be a non-empty column vector with shape {n_samples, 1}");
    }

    std::map<double, int> class_counts;

    for (int i = 0; i < y.rows(); i++) {
        class_counts[y.at(i, 0)]++;
    }

    double best_label = 0.0;
    int best_count = -1;

    for (const auto &[label, count] : class_counts) {
        if (count > best_count) {
            best_label = label;
            best_count = count;
        }
    }

    return best_label;
}

}

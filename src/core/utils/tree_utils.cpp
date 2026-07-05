#include "core/utils/tree_utils.hpp"

#include <map>
#include <stdexcept>

double gini_impurity(const Matrix &y) {
    if (y.rows() == 0 || y.cols() != 1) {
        throw std::invalid_argument("y must be a non-empty column vector");
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


double majority_class(const Matrix &y) {
    if (y.rows() == 0 || y.cols() != 1) {
        throw std::invalid_argument("y must be a non-empty column vector");
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